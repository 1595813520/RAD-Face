import cv2
from insightface.app import FaceAnalysis
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL
from PIL import Image
from ip_adapter.ip_adapter_faceid import IPRetrievalFaceID  # 假设 IPRetrievalFaceID 在此模块中
from utils.retrieval_utils import FaissRetriever

# 初始化 InsightFace 模型
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(512, 512))

# 读取输入图像并提取面部嵌入
image = cv2.imread("/data/IP-Adapter/image/sam_blurred.png")
faces = app.get(image)
faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)  # [1, 512]

# 模型路径和参数
base_model_path = "/data/IP-Adapter/models/Realistic_Vision_V4.0_noVAE"
vae_model_path = "/data/model/sd-vae-ft-mse"
ip_ckpt = "/data/IP-Adapter/models/ip-retrieval.bin"
device = "cuda"
torch_dtype = torch.float16

# 初始化调度器和 VAE
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch_dtype, device=device)

# 加载 Stable Diffusion 管道
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch_dtype,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# 初始化 FaissRetriever
retriever = FaissRetriever(
    index_path="/data/IP-retrieval/dataset/faiss/hq/face_index.faiss",
    features_path="/data/IP-retrieval/dataset/faiss/hq/all_features.npy",
    key_mapping_path="/data/IP-retrieval/dataset/faiss/hq/face_key_mapping.json",
)

# 检索 top-k 个特征
top_k = 5
_, _, retrieval_features_set = retriever.retrieve_top_k(faceid_embeds, k=top_k)  # [1, k, 512]

# 实例化 IP-Adapter 模型
ip_model = IPRetrievalFaceID(
    sd_pipe=pipe,
    ip_ckpt=ip_ckpt,
    retriever=retriever,
    device=device,
    lora_rank=128,
    num_tokens=4,
    torch_dtype=torch.float16
)

# 生成图像
prompt = "photo of a man in black jacket in a garden"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

images = ip_model.generate(
    faceid_embeds=faceid_embeds,
    retrieval_features_set=retrieval_features_set,  # 使用 retrieval_features_set
    prompt=prompt,
    negative_prompt=negative_prompt,
    scale=1.0,
    num_samples=4,
    width=512,
    height=512,
    num_inference_steps=30,
    seed=2023,
    guidance_scale=7.5
)

# 保存生成的图像
for i, img in enumerate(images):
    img_filename = f"/data/IP-Adapter/image/sam_blur_{i+1}.png"
    img.save(img_filename)
    print(f"Image {i+1} saved as {img_filename}")