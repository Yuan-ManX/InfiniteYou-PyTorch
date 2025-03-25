import math
import os
import random
from typing import Optional
import cv2
import numpy as np
import torch
from diffusers.models import FluxControlNetModel
from facexlib.recognition import init_recognition_model
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image

from flux_infusenet import FluxInfuseNetPipeline
from resampler import Resampler


# 设置随机种子以确保结果可复现
def seed_everything(seed, deterministic=False):
    """
    设置全局随机种子，以确保实验的可重复性。

    参数:
        seed (int): 要设置的随机种子。
        deterministic (bool, optional): 是否设置CUDNN后端的确定性选项。
            如果设置为True，则将 `torch.backends.cudnn.deterministic` 设置为True，
            并将 `torch.backends.cudnn.benchmark` 设置为False。默认值为False。
    """
    # 设置Python内置的随机数种子
    random.seed(seed)
    # 设置NumPy的随机数种子
    np.random.seed(seed)
    # 设置CPU上的PyTorch随机数种子
    torch.manual_seed(seed)
    # 设置当前GPU的PyTorch随机数种子
    torch.cuda.manual_seed(seed)
    # 设置所有GPU的PyTorch随机数种子
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    """
    在PIL图像上绘制关键点及其连接线。

    参数:
        image_pil (PIL.Image.Image): 输入的PIL图像。
        kps (list或numpy.ndarray): 关键点列表，每个关键点包含 (x, y) 坐标。
        color_list (list, optional): 颜色列表，用于绘制不同的连接线和关键点。
            默认值为 [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]。

    返回:
        PIL.Image.Image: 绘制了关键点及其连接线的图像。
    """
    # 连接线的宽度
    stickwidth = 4
    # 定义连接的关键点对
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    # 将关键点转换为NumPy数组
    kps = np.array(kps)

    # 获取图像的宽度和高度
    w, h = image_pil.size
    # 创建一个与图像大小相同的黑色背景图像
    out_img = np.zeros([h, w, 3])

    # 绘制连接线
    for i in range(len(limbSeq)):
        # 获取当前连接的关键点索引对
        index = limbSeq[i]
        # 根据关键点索引选择颜色
        color = color_list[index[0]]

        # 获取连接线的x坐标
        x = kps[index][:, 0]
        # 获取连接线的y坐标
        y = kps[index][:, 1]
        # 计算连接线的长度
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        # 计算连接线的角度
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        # 计算连接线的多边形表示，用于绘制椭圆
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        # 填充连接线多边形
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    # 调整连接线的透明度
    out_img = (out_img * 0.6).astype(np.uint8)
    
    # 绘制关键点
    for idx_kp, kp in enumerate(kps):
        # 选择关键点的颜色
        color = color_list[idx_kp]
        # 获取关键点的坐标
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def extract_arcface_bgr_embedding(in_image, landmark, arcface_model=None, in_settings=None):
    """
    使用ArcFace模型提取人脸的BGR嵌入。

    参数:
        in_image (numpy.ndarray): 输入的BGR图像。
        landmark (list或numpy.ndarray): 人脸的关键点坐标。
        arcface_model (torch.nn.Module, optional): 预训练的ArcFace模型。如果为None，则初始化一个模型。默认值为None。
        in_settings (dict, optional): 设置参数。如果需要，可以传入额外的设置。默认值为None。

    返回:
        torch.Tensor: 提取的人脸嵌入向量，形状为 [512]。
    """
    # 获取关键点
    kps = landmark
    # 使用关键点对人脸进行归一化裁剪，输出图像大小为112x112
    arc_face_image = face_align.norm_crop(in_image, landmark=np.array(kps), image_size=112)
    # 将图像转换为PyTorch张量，并进行归一化处理
    arc_face_image = torch.from_numpy(arc_face_image).unsqueeze(0).permute(0,3,1,2) / 255.
    # 将像素值从 [0,1] 缩放到 [-1, 1]
    arc_face_image = 2 * arc_face_image - 1
    # 将张量移动到GPU并保证内存连续
    arc_face_image = arc_face_image.cuda().contiguous()
    if arcface_model is None:
        arcface_model = init_recognition_model('arcface', device='cuda')
    # 提取人脸嵌入向量
    face_emb = arcface_model(arc_face_image)[0] # [512], normalized
    # 返回人脸嵌入向量
    return face_emb


def resize_and_pad_image(source_img, target_img_size):
    """
    调整图像大小并填充以适应目标尺寸，保持图像的纵横比。

    参数:
        source_img (PIL.Image.Image): 原始图像。
        target_img_size (tuple): 目标图像尺寸 (宽度, 高度)。

    返回:
        PIL.Image.Image: 调整大小并填充后的图像。
    """
    # 获取原始图像和目标图像的尺寸
    source_img_size = source_img.size
    target_width, target_height = target_img_size
    
    # 根据目标图像的较短边确定新尺寸，保持原始图像的纵横比
    if target_width <= target_height:
        new_width = target_width
        new_height = int(target_width * (source_img_size[1] / source_img_size[0]))
    else:
        new_height = target_height
        new_width = int(target_height * (source_img_size[0] / source_img_size[1]))
    
    # 使用高质量的LANCZOS插值方法调整图像大小
    resized_source_img = source_img.resize((new_width, new_height), Image.LANCZOS)
    
    # 计算填充量以使调整后的图像居中
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    
    # 创建一个白色背景的新图像
    padded_img = Image.new("RGB", target_img_size, (255, 255, 255))
    # 将调整后的图像粘贴到新图像上
    padded_img.paste(resized_source_img, (pad_left, pad_top))
    
    return padded_img


class InfUFluxPipeline:
    def __init__(
            self, 
            base_model_path, 
            infu_model_path, 
            insightface_root_path = './',
            image_proj_num_tokens=8,
            infu_flux_version='v1.0',
            model_version='aes_stage2',
        ):
        """
        InfUFluxPipeline 类的初始化方法，用于加载模型和必要的组件。

        参数:
            base_model_path (str): 基础模型的路径。
            infu_model_path (str): InfiniteYou模型的路径。
            insightface_root_path (str, optional): InsightFace模型的根路径，默认为当前目录。
            image_proj_num_tokens (int, optional): 图像投影模型的token数量，默认为8。
            infu_flux_version (str, optional): InfiniteYou模型的版本，默认为v1.0。
            model_version (str, optional): 模型的具体版本，默认为aes_stage2。
        """

        self.infu_flux_version = infu_flux_version
        self.model_version = model_version
        
        # 加载 InfiniteYou 的控制网络模型
        try:
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
        except:
            print("No InfiniteYou model found. Downloading from HuggingFace `ByteDance/InfiniteYou` to `./models/InfiniteYou` ...")
            snapshot_download(repo_id='ByteDance/InfiniteYou', local_dir='./models/InfiniteYou', local_dir_use_symlinks=False)
            infu_model_path = os.path.join('./models/InfiniteYou', f'infu_flux_{infu_flux_version}', model_version)
            infusenet_path = os.path.join(infu_model_path, 'InfuseNetModel')
            self.infusenet = FluxControlNetModel.from_pretrained(infusenet_path, torch_dtype=torch.bfloat16)
            insightface_root_path = './models/InfiniteYou/supports/insightface'
        
        # 加载基础管道模型
        try:
            pipe = FluxInfuseNetPipeline.from_pretrained(
                base_model_path,
                controlnet=self.infusenet,
                torch_dtype=torch.bfloat16,
            )
        except:
            try:
                pipe = FluxInfuseNetPipeline.from_single_file(
                    base_model_path,
                    controlnet=self.infusenet,
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                print(e)
                print('\nIf you are using `black-forest-labs/FLUX.1-dev` and have not downloaded it into a local directory, '
                      'please accept the agreement and obtain access at https://huggingface.co/black-forest-labs/FLUX.1-dev. '
                      'Then, use `huggingface-cli login` and your access tokens at https://huggingface.co/settings/tokens to authenticate. '
                      'After that, run the code again. If you have downloaded it, please use `base_model_path` to specify the correct path.')
                print('\nIf you are using other models, please download them to a local directory and use `base_model_path` to specify the correct path.')
                exit()
        pipe.to('cuda', torch.bfloat16)
        self.pipe = pipe

        # 加载图像投影模型
        # 获取图像投影模型的token数量
        num_tokens = image_proj_num_tokens
        # 图像嵌入的维度，默认为512
        image_emb_dim = 512
        # 初始化Resampler模型
        image_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=num_tokens,
            embedding_dim=image_emb_dim,
            output_dim=4096,
            ff_mult=4,
        )
        # 构建图像投影模型的路径
        image_proj_model_path = os.path.join(infu_model_path, 'image_proj_model.bin')
        # 从CPU加载模型状态字典
        ipm_state_dict = torch.load(image_proj_model_path, map_location="cpu")
        image_proj_model.load_state_dict(ipm_state_dict['image_proj'])
        del ipm_state_dict
        image_proj_model.to('cuda', torch.bfloat16)
        image_proj_model.eval()

        self.image_proj_model = image_proj_model

        # 加载人脸编码器
        # 初始化640x640分辨率的人脸分析模型
        self.app_640 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 准备模型，检测尺寸为640x640
        self.app_640.prepare(ctx_id=0, det_size=(640, 640))

        # 初始化320x320分辨率的人脸分析模型
        self.app_320 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 准备模型，检测尺寸为320x320
        self.app_320.prepare(ctx_id=0, det_size=(320, 320))

        # 初始化160x160分辨率的人脸分析模型
        self.app_160 = FaceAnalysis(name='antelopev2', 
                                root=insightface_root_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # 准备模型，检测尺寸为160x160
        self.app_160.prepare(ctx_id=0, det_size=(160, 160))

        # 初始化ArcFace人脸识别模型，并移动到GPU
        self.arcface_model = init_recognition_model('arcface', device='cuda')

    def load_loras(self, loras):
        """
        加载LoRA（低秩适应）权重。

        参数:
            loras (list): 包含LoRA路径、名称和缩放比例的列表。
        """
        # 初始化名称和缩放比例列表
        names, scales = [],[]
        for lora_path, lora_name, lora_scale in loras:
            if lora_path != "":
                print(f"loading lora {lora_path}")
                self.pipe.load_lora_weights(lora_path, adapter_name = lora_name)
                names.append(lora_name)
                scales.append(lora_scale)

        if len(names) > 0:
            self.pipe.set_adapters(names, adapter_weights=scales)

    def _detect_face(self, id_image_cv2):
        """
        检测图像中的人脸。

        参数:
            id_image_cv2 (numpy.ndarray): 输入的BGR格式图像。

        返回:
            list: 检测到的人脸信息列表。
        """
        # 使用640x640模型检测人脸
        face_info = self.app_640.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info
        
        face_info = self.app_320.get(id_image_cv2)
        if len(face_info) > 0:
            return face_info

        face_info = self.app_160.get(id_image_cv2)
        return face_info

    def __call__(
        self,
        id_image: Image.Image,  # PIL.Image.Image (RGB)
        prompt: str,
        control_image: Optional[Image.Image] = None,  # PIL.Image.Image (RGB) or None
        width = 864,
        height = 1152,
        seed = 42,
        guidance_scale = 3.5,
        num_steps = 30,
        infusenet_conditioning_scale = 1.0,
        infusenet_guidance_start = 0.0,
        infusenet_guidance_end = 1.0,
    ):        
        """
        前向传播方法，用于生成图像。

        参数:
            id_image (PIL.Image.Image): 输入的ID图像，PIL.Image.Image (RGB)。
            prompt (str): 提示文本。
            control_image (PIL.Image.Image, optional): 控制图像，PIL.Image.Image (RGB) 或 None。
            width (int, optional): 输出图像宽度，默认为864。
            height (int, optional): 输出图像高度，默认为1152。
            seed (int, optional): 随机种子，默认为42。
            guidance_scale (float, optional): 指导尺度，默认为3.5。
            num_steps (int, optional): 推理步数，默认为30。
            infusenet_conditioning_scale (float, optional): InfuseNet条件缩放比例，默认为1.0。
            infusenet_guidance_start (float, optional): InfuseNet指导开始比例，默认为0.0。
            infusenet_guidance_end (float, optional): InfuseNet指导结束比例，默认为1.0。

        返回:
            PIL.Image.Image: 生成的图像。
        """
        # 提取ID嵌入
        print('Preparing ID embeddings')
        # 将PIL图像转换为OpenCV BGR格式
        id_image_cv2 = cv2.cvtColor(np.array(id_image), cv2.COLOR_RGB2BGR)
        # 检测人脸
        face_info = self._detect_face(id_image_cv2)
        if len(face_info) == 0:
            raise ValueError('No face detected in the input ID image')
        
        # 选择最大的脸
        face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
        # 获取关键点
        landmark = face_info['kps']
        # 提取ArcFace嵌入
        id_embed = extract_arcface_bgr_embedding(id_image_cv2, landmark, self.arcface_model)
        id_embed = id_embed.clone().unsqueeze(0).float().cuda()
        # 重塑形状
        id_embed = id_embed.reshape([1, -1, 512])
        # 转换为bfloat16精度
        id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            # 通过图像投影模型处理
            id_embed = self.image_proj_model(id_embed)
            # 获取形状
            bs_embed, seq_len, _ = id_embed.shape
            # 重复张量
            id_embed = id_embed.repeat(1, 1, 1)
            # 重塑形状
            id_embed = id_embed.view(bs_embed * 1, seq_len, -1)
            # 转换为bfloat16精度
            id_embed = id_embed.to(device='cuda', dtype=torch.bfloat16)
        
        # 加载控制图像
        print('Preparing the control image')
        if control_image is not None:
            control_image = control_image.convert("RGB")
            # 调整大小并填充
            control_image = resize_and_pad_image(control_image, (width, height))
            # 检测人脸
            face_info = self._detect_face(cv2.cvtColor(np.array(control_image), cv2.COLOR_RGB2BGR))
            if len(face_info) == 0:
                raise ValueError('No face detected in the control image')
            # 选择最大的脸
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
            # 绘制关键点
            control_image = draw_kps(control_image, face_info['kps'])
        else:
            # 创建黑色图像
            out_img = np.zeros([height, width, 3])
            # 转换为PIL图像
            control_image = Image.fromarray(out_img.astype(np.uint8))

        # 执行推理
        print('Generating image')
        # 设置随机种子
        seed_everything(seed)
        # 调用管道模型生成图像
        image = self.pipe(
            prompt=prompt,
            controlnet_prompt_embeds=id_embed, # 使用ID嵌入作为控制net提示嵌入
            control_image=control_image, # 使用控制图像
            guidance_scale=guidance_scale, # 设置指导尺度
            num_inference_steps=num_steps, # 设置推理步数
            controlnet_guidance_scale=1.0, # 设置控制net指导尺度
            controlnet_conditioning_scale=infusenet_conditioning_scale, # 设置控制net条件缩放比例
            control_guidance_start=infusenet_guidance_start, # 设置控制指导开始比例
            control_guidance_end=infusenet_guidance_end, # 设置控制指导结束比例
            height=height,
            width=width,
        ).images[0]

        return image
