import torch
import soundfile as sf
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class QwenModel:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B", device="auto", use_flash_attn=False):
        self.device = device
        self.use_audio_in_video = True
        
        # 加载模型
        if use_flash_attn:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device,
                attn_implementation="flash_attention_2",
            )
        else:
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=device,
            )
        
        # 不需要音频输出时可以禁用talker，节省GPU内存
        self.model.disable_talker()
        
        # 加载处理器
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    
    def process_video(self, video_path):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that transcribes and summarizes video content."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Please transcribe all speech in this video and describe its content."}
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=self.use_audio_in_video)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=self.use_audio_in_video
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        text_ids = self.model.generate(**inputs, use_audio_in_video=self.use_audio_in_video, return_audio=False)
        result = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return result[0] if result else ""
    
    def process_image(self, image_path):
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that describes image content."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Please describe this image in detail, including any text, people, objects, and context."}
                ],
            },
        ]
        
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=False
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        
        text_ids = self.model.generate(**inputs, use_audio_in_video=False, return_audio=False)
        result = self.processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return result[0] if result else ""