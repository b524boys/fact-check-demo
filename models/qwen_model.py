import torch
import os
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

class QwenModel:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B", device="auto", cache_dir=None):
        self.device = device
        self.use_audio_in_video = True
        self.cache_dir = cache_dir or "/home/featurize/data/models"
        
        print(f"Loading Qwen2.5-Omni model from {model_path}")
        print(f"Model will be downloaded and cached in: {self.cache_dir}")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                print(f"Created directory: {self.cache_dir}")
            except OSError as e:
                print(f"Failed to create directory {self.cache_dir}: {e}")
                print("Please create this directory manually and ensure write permissions.")
        
        try:
            # Load the model
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=device,
                cache_dir=self.cache_dir
            )
            
            # Disable talker to save GPU memory
            self.model.disable_talker()
            print("Talker module disabled to save memory.")
            
            # Load the processor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                model_path,
                cache_dir=self.cache_dir
            )
            print("Model and processor loaded successfully.")
        except ImportError:
            print("Error loading model: Flash attention may not be properly installed.")
            print("If you don't need Flash Attention 2, comment out the attn_implementation='flash_attention_2' parameter in from_pretrained.")
            print("Or try installing: pip install -U flash-attn --no-build-isolation")
            raise
        except Exception as e:
            print(f"Error loading model or processor: {e}")
            print("Please ensure all dependencies are properly installed and network connection is stable.")
            raise
    
    def process_video_visual(self, video_path):
        """Process a video and return its visual content description only"""
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that describes visual content in videos. Focus ONLY on describing what you see, not what you hear."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Please describe ONLY the visual content of this video in detail, including scenes, people, objects, actions, and any text that appears. Do NOT include any audio description or speech transcription."}
                ],
            },
        ]
        
        return self._generate_response(conversation, use_audio=False)
    
    def process_video_audio(self, video_path):
        """Process a video and return its audio content description only"""
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that transcribes speech and describes audio content. Focus ONLY on what you hear, not what you see."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": "Please transcribe ONLY the speech and describe ONLY the audio content in this video, such as music, sound effects, and background noise. Do NOT include any descriptions of visual elements."}
                ],
            },
        ]
        
        return self._generate_response(conversation, use_audio=True)
    
    def process_video(self, video_path):
        """Process a video and return both visual and audio content descriptions"""
        visual_content = self.process_video_visual(video_path)
        audio_content = self.process_video_audio(video_path)
        
        return {
            "visual_content": visual_content,
            "audio_content": audio_content,
            "full_content": f"VISUAL CONTENT:\n{visual_content}\n\nAUDIO CONTENT:\n{audio_content}"
        }
    
    def process_image(self, image_path):
        """Process an image and return its content description"""
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant that describes image content comprehensively and accurately."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": "Please describe this image in detail, including any text, people, objects, colors, composition, and context. Be comprehensive and focus on all visible elements."}
                ],
            },
        ]
        
        return self._generate_response(conversation, use_audio=False)
    
    def verify_claim(self, claim, media_path, media_type='image'):
        """Verify if a claim is true, false, or unrelated to the media content"""
        system_prompt = (
            "You are a rigorous fact-checking assistant. Your task is to analyze a statement and a multimedia file "
            "(which may be an image or a video). You need to carefully analyze the multimedia content. "
            "Then, based on the information presented in the multimedia file, judge whether the provided statement "
            "is 'correct', 'incorrect', or 'unrelated' to the file's content. "
            "Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'. "
            "Do not add any additional explanation or information."
        )
        
        user_content = [
            {"type": "text", "text": f"Statement: {claim}"}
        ]
        
        if media_type == 'image':
            user_content.append({"type": "image", "image": media_path})
        elif media_type == 'video':
            user_content.append({"type": "video", "video": media_path})
            
        user_content.append({"type": "text", "text": "Please evaluate the above statement based on the provided multimedia file."})
        
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        
        use_audio = self.use_audio_in_video if media_type == 'video' else False
        return self._generate_response(conversation, use_audio=use_audio)
    
    def _generate_response(self, conversation, use_audio=False):
        """Generate a response based on the conversation"""
        try:
            # Prepare text prompt for processor
            text_prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # Process multimedia information
            audios, images, videos = process_mm_info(
                conversation,
                use_audio_in_video=use_audio
            )
            
            # Process inputs
            inputs = self.processor(
                text=text_prompt,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio
            )
            
            # Move inputs to the model's device
            inputs = {k: v.to(self.model.device).to(v.dtype if hasattr(v, 'dtype') and v.dtype.is_floating_point else v.dtype) for k, v in inputs.items()}
            
            # Generate response
            generated_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                return_audio=False,
                max_new_tokens=512
            )
            
            # Decode response
            response_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            
            if response_text and isinstance(response_text, list):
                return response_text[0].strip()
            elif isinstance(response_text, str):
                return response_text.strip()
            else:
                return "Failed to get a valid response."
                
        except Exception as e:
            print(f"Error during model inference or decoding: {e}")
            return f"Error processing request: {str(e)}"