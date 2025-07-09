import os
import io
import base64
import requests
from openai import OpenAI
from PIL import Image
import time


# 全局参数配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, "CFTrain")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "output")
STABLE_DIFFUSION_URL = "http://127.0.0.1:7860"
API_KEY = "sk-4bnlGhpyBVlJN5TFmz6V4dpy4yAXEZbEk22nn4kp3IcdSEGn"

# 基本生成参数
MODEL_NAME = "landscapeRealistic_v20WarmColor.safetensors [ca2e3bd9f9]"       # Stable Diffusion模型
POSITIVE_PROMPT = "bird-eye,ultra-realistic satellite image,WorldView-3 style,sharp details, \
                   <lora:'Improved Amateur Snapshot Photo Realism' - v12 - [STYLE] [LORA] [FLUX] - spectrum_0001 by 'AI_Characters':1>,"
NEGATIVE_PROMPT = "(worst quality, low quality, normal quality:2),( easynegative:1.5), \
                    blurry,overexposed,oversaturated,cartoon,CGI, \
                    cartoon,illustration,painting,oversaturated colors,fantasy scenery, \
                    people,animals,buildings,watermark,text,logo,frame,"  

# 图生图参数
SAMPLER_NAME = "DPM++ 2M"  # 采样方法
SCHEDULER_TYPE = "Karras"  # 调度类型
STEPS = 30                 # 步数
WIDTH = 1024               # 输出图像宽度
HEIGHT = 1024              # 输出图像高度
CFG_SCALE = 7              # CFG提示相关度系数
DENOISING_STRENGTH = 0.6   # 重绘幅度
SEED = -1                  # 随机种子
BATCH_SIZE = 2             # 生成图片数量

# ControlNet参数
CONTROLNET_MODEL = "control_v11p_sd15_canny [d14c016b]"     # ControlNet模型
PIXEL_PERFECT = True                                        # 完美像素模式
CONTROLNET_MODULE = "canny"                                 # 预处理器
CONTROLNET_WEIGHT = 1.0                                     # 控制权重
GUIDANCE_START = 0.0                                        # 引导介入时机
GUIDANCE_END = 1.0                                          # 引导终止时机
CONTROLNET_THRESHOLD_A = 90                                 # 低阈值
CONTROLNET_THRESHOLD_B = 180                                # 高阈值
CONTROL_MODE = "ControlNet is more important"               # 控制模式 

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_prompt_for_image(image_path, output_dir):
    """
    调用OpenAI GPT-4 API，分析图像并生成描述性提示词。
    返回生成的图片内容文本字符串。
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(output_dir, "prompt", f"{base_name}.txt")

    # 如果存在txt文件，则直接读取
    if os.path.exists(txt_path):
        prompt_text = "FLAG" # 如果存在txt文件，则直接返回FLAG，用于跳过生成
        # with open(txt_path, "r", encoding="utf-8") as txt_file:
        #     prompt_text = txt_file.read()
        return prompt_text
    
    # 将图像读取为base64字符串
    with open(image_path, "rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{img_b64}"

    # 构建 GPT-4 请求消息，提示GPT描述图像内容
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key=API_KEY,
    )
    max_retries = 5  # 最大重试次数
    retry_delay = 2  # 重试间隔
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": 
                            "请根据stable diffusion的正向提示词格式，给我描述图片内容的正向提示词，不要有任何其他内容。\n"
                            },
                            {"type": "image_url", "image_url": {"url": data_url}
                            },
                        ],
                    }
                ],
                timeout=30  # 设置超时时间为 30 秒
            )            
            prompt_text = response.choices[0].message.content.strip()
            
            if not prompt_text:
                print("返回内容为空，重试...")
                continue
            if any('\u4e00' <= char <= '\u9fff' for char in prompt_text):
                print(prompt_text)
                print("返回内容包含中文，重试...")
                continue
            special_chars = set('!@#$%^&*()_+{}[]|\\:;"\'<>?/~`')
            if any(char in special_chars for char in prompt_text):
                print(prompt_text)
                print("返回内容包含特殊字符，重试...")
                continue
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"请求失败 ({str(e)})，{retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                raise Exception(f"达到最大重试次数 ({max_retries})，最后一次错误: {str(e)}")

    # 将提示词保存为与图像同名的txt文件
    with open(txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(prompt_text)
    return prompt_text

def generate_images_with_controlnet(image_path, prompt_text):
    """
    调用Stable Diffusion的img2img接口，结合ControlNet（Canny）生成图像变体。
    返回生成的PIL图像对象列表。
    """
    # 读取源图像并编码为Base64
    with open(image_path, "rb") as f:
        img_data = f.read()
    img_b64 = base64.b64encode(img_data).decode('utf-8')

    # 请求payload
    payload = {
        "init_images": [img_b64],
        "prompt": POSITIVE_PROMPT + prompt_text,
        "negative_prompt": NEGATIVE_PROMPT,
        "width": WIDTH,
        "height": HEIGHT,
        "sampler_name": SAMPLER_NAME,
        "scheduler_name": SCHEDULER_TYPE,
        "steps": STEPS,
        "cfg_scale": CFG_SCALE,
        "denoising_strength": DENOISING_STRENGTH,
        "seed": SEED,
        "n_iter": 1,
        "batch_size": BATCH_SIZE,

        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "image": img_b64,        # 不上传独立的控制图像
                        "module": CONTROLNET_MODULE,
                        "model": CONTROLNET_MODEL,
                        "weight": 1.0,
                        "threshold_a": CONTROLNET_THRESHOLD_A,
                        "threshold_b": CONTROLNET_THRESHOLD_B,
                        "resize_mode": "Just Resize",       # 仅调整大小
                        "guidance_start": GUIDANCE_START,
                        "guidance_end": GUIDANCE_END,
                        "control_mode": CONTROL_MODE,
                        "pixel_perfect": PIXEL_PERFECT,
                        "save_detected_map": False,  # 不保存 Canny 线稿
                    }
                ]
            }
        }
    }

    if MODEL_NAME:
        payload["override_settings"] = {"sd_model_checkpoint": MODEL_NAME}
        payload["override_settings_restore_afterwards"] = True

    # 发送POST请求
    url = f"{STABLE_DIFFUSION_URL}/sdapi/v1/img2img"
    response = requests.post(url, json=payload)
    response_data = response.json()
    images_data = response_data.get("images", [])
    result_images = []
    for img_str in images_data:
        # 返回的图像base64可能包含前缀
        if img_str.startswith("data:image"):
            img_str = img_str.split(",", 1)[1]
        img_bytes = base64.b64decode(img_str)
        # 使用PIL打开图像字节数据
        img = Image.open(io.BytesIO(img_bytes)) if 'io' in globals() else Image.open(io.BytesIO(img_bytes))
        result_images.append(img)
    return result_images

def main(image_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            image_path = os.path.join(image_dir, filename)
            print(f"处理图像: {image_path}")

            # 1.生成提示词
            prompt = generate_prompt_for_image(image_path, output_dir)
            if prompt == "FLAG":
                continue
            print(f"GPT-4生成的提示词: {prompt}")

            # 2.生成图像
            images = generate_images_with_controlnet(image_path, prompt)

            # 3.保存
            base_name = os.path.splitext(filename)[0]
            for idx, img in enumerate(images):
                out_path = os.path.join(output_dir, f"{base_name}_gen{idx+1}.png")
                img.save(out_path)
                print(f"生成图像已保存: {out_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    subdir = [name for name in os.listdir(IMAGE_DIR) if os.path.isdir(os.path.join(IMAGE_DIR, name))]
    for sub in subdir:
        if not os.path.exists(os.path.join(OUTPUT_DIR, sub)):
            os.makedirs(os.path.join(OUTPUT_DIR, sub))
        if not os.path.exists(os.path.join(OUTPUT_DIR, sub, "prompt")):
            os.makedirs(os.path.join(OUTPUT_DIR, sub, "prompt"))

        main(os.path.join(IMAGE_DIR, sub), os.path.join(OUTPUT_DIR, sub))