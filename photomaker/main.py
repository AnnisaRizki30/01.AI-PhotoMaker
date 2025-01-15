import sys
import torch
from torch.cuda.amp import autocast
import cv2
import numpy as np
import random
import os
import gradio as gr
from PIL import Image
from diffusers.utils import load_image
from diffusers import DDIMScheduler
from huggingface_hub import hf_hub_download
from photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline
from gradio_demo.style_template import styles

from torchvision.transforms import functional
sys.modules["torchvision.transforms.functional_tensor"] = functional

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=SyntaxWarning)


# global variable
base_model_path = 'SG161222/RealVisXL_V3.0'
device = "cuda" if torch.cuda.is_available() else "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Photographic (Default)"

# download PhotoMaker checkpoint to cache
photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")

pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)

pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_ckpt),
    subfolder="",
    weight_name=os.path.basename(photomaker_ckpt),
    trigger_word="img"
)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.set_adapters(["photomaker"], adapter_weights=[1.0])
pipe.fuse_lora()

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative

def resize_image(image, max_dim=1000):
    h, w = image.shape[:2]
    scale_factor = max_dim / max(h, w)
    if scale_factor < 1:
        new_size = (int(w * scale_factor), int(h * scale_factor))
        image_resized = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        return image_resized
    return image

def enhance_image(img_array):
    try:
        model_path = 'realesr-general-x4v3.pth'
        half = True if torch.cuda.is_available() else False
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        upsampler = RealESRGANer(scale=1, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half) 

        if isinstance(img_array, Image.Image):
            img_array = np.array(img_array)
        img_array = resize_image(img_array, max_dim=1000)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        with autocast():
            enhanced_image, _ = upsampler.enhance(img_array, outscale=1)

        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        return enhanced_image
    except Exception as error:
        print('Enhancement failed:', error)
        return None
      

def generate_image(upload_images, prompt, style_name):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    prompt, negative_prompt = apply_style(style_name, prompt)

    num_steps = 20
    style_strength_ratio = 30
    guidance_scale = 7.5
    seed = 1930231864

    if upload_images is None:
        return "Please upload an image."

    input_image = upload_images
    generator = torch.Generator(device=device).manual_seed(seed)

    print("Start inference...")
    print(f"[Debug] Prompt: {prompt}, \n[Debug] Neg Prompt: {negative_prompt}")
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    print(start_merge_step)

    with autocast():
        images = pipe(
            prompt=prompt,
            input_id_images=[input_image],
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            num_inference_steps=num_steps,
            start_merge_step=start_merge_step,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images

    print("Upscaling the image...")
    output_image = images[0]
    final_image = enhance_image(np.array(output_image))

    if final_image is None:
        print("Upscaling failed. Returning original image.")
        return output_image

    enhance_image_pil = Image.fromarray(final_image)
    return enhance_image_pil
