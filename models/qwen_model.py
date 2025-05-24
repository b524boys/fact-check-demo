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
    
    def verify_claim(self, claim, media_path, media_type='image', enable_deepfake_detection=False):
        """Verify if a claim is true, false, or unrelated to the media content with deepfake detection"""
        
        if enable_deepfake_detection:
            if media_type == 'image':
                system_prompt = self._get_image_verification_system_prompt()
                user_prompt = self._get_image_verification_user_prompt(claim)
            elif media_type == 'video':
                system_prompt = self._get_video_verification_system_prompt()
                user_prompt = self._get_video_verification_user_prompt(claim)
            else:
                system_prompt = self._get_basic_verification_system_prompt()
                user_prompt = self._get_basic_verification_user_prompt(claim)
        else:
            if media_type == 'image':
                system_prompt = self._get_image_verification_system_prompt_standard()
                user_prompt = self._get_image_verification_user_prompt_standard(claim)
            elif media_type == 'video':
                system_prompt = self._get_video_verification_system_prompt_standard()
                user_prompt = self._get_video_verification_user_prompt_standard(claim)
            else:
                system_prompt = self._get_basic_verification_system_prompt_standard()
                user_prompt = self._get_basic_verification_user_prompt_standard(claim)
        
        user_content = [
            {"type": "text", "text": f"Statement: {claim}"}
        ]
        
        if media_type == 'image':
            user_content.append({"type": "image", "image": media_path})
        elif media_type == 'video':
            user_content.append({"type": "video", "video": media_path})
            
        user_content.append({"type": "text", "text": user_prompt})
        
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
    
    def _get_image_verification_system_prompt_with_deepfake(self):
        """Get system prompt for image verification with deepfake detection"""
        return """You are a rigorous fact-checking assistant with specialized training in detecting manipulated media content. Your task is to analyze a statement and an image, then determine if the statement is 'correct', 'incorrect', or 'unrelated' to the image content.

CRITICAL ANALYSIS REQUIREMENTS:

1. **DEEPFAKE AND MANIPULATION DETECTION:**
   - Carefully examine facial features, especially around the eyes, mouth, nose, and ears
   - Look for inconsistencies in skin texture, lighting, and shadows on faces
   - Check for unnatural blending or artifacts around facial boundaries
   - Pay attention to asymmetrical facial features that seem inconsistent
   - Notice any unusual pixel patterns, blurring, or digital artifacts around faces
   - Observe if facial expressions match the context and body language
   - Check for inconsistent aging, hair color, or facial structure
   - Look for mismatched lighting between the face and the rest of the image

2. **TECHNICAL MANIPULATION INDICATORS:**
   - Examine compression artifacts and inconsistent image quality
   - Look for copy-paste artifacts or duplicated elements
   - Check for inconsistent shadows, reflections, and lighting
   - Notice any warping or distortion around edited areas
   - Identify unusual color gradients or saturation inconsistencies

3. **CONTEXTUAL ANALYSIS:**
   - Verify if the background and setting are consistent with the claimed scenario
   - Check if clothing, objects, and environment match the timeframe and context
   - Look for anachronisms or contextually impossible elements

4. **VERIFICATION PROCESS:**
   - If you detect potential manipulation or deepfake indicators, note them specifically
   - Consider the overall credibility of the image based on technical analysis
   - Factor in manipulation likelihood when evaluating the statement

Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'.

**IMPORTANT:** If you detect clear signs of deepfake or significant manipulation that affects the claim's validity, lean towards 'incorrect' unless the statement explicitly acknowledges the manipulated nature."""
    
    def _get_image_verification_user_prompt_with_deepfake(self, claim):
        """Get user prompt for image verification"""
        return f"""Please analyze the provided image with special attention to potential deepfakes or digital manipulations, then evaluate the statement: "{claim}"

ANALYSIS CHECKLIST:
1. Examine all facial features for deepfake indicators
2. Check for technical manipulation signs
3. Verify contextual consistency
4. Determine the relationship between the statement and the authentic visual content

Provide your judgment: 'correct', 'incorrect', or 'unrelated'."""
    
    def _get_video_verification_system_prompt(self):
        """Get system prompt for video verification with deepfake detection"""
        return """You are a rigorous fact-checking assistant with specialized training in detecting manipulated video content and deepfakes. Your task is to analyze a statement and a video, then determine if the statement is 'correct', 'incorrect', or 'unrelated' to the video content.

CRITICAL ANALYSIS REQUIREMENTS:

1. **VIDEO DEEPFAKE DETECTION:**
   - Examine facial consistency across different frames
   - Look for temporal inconsistencies in facial features, especially around eyes, mouth, and jawline
   - Notice unnatural facial movements or expressions that don't match speech patterns
   - Check for flickering or instability around facial boundaries between frames
   - Observe lip-sync accuracy with audio content
   - Look for frame-to-frame inconsistencies in lighting on faces
   - Notice any unusual blurring or artifacts that appear and disappear

2. **TECHNICAL VIDEO MANIPULATION INDICATORS:**
   - Check for compression artifacts that vary across the video
   - Look for frame rate inconsistencies or stuttering
   - Notice color and lighting inconsistencies between cuts or scenes
   - Identify any obvious editing seams or transitions
   - Check for audio-visual synchronization issues

3. **MOTION AND BEHAVIOR ANALYSIS:**
   - Verify if body language matches facial expressions
   - Check if head movements appear natural and consistent
   - Look for unnatural eye movements or blinking patterns
   - Notice if gestures match the speaking rhythm and content

4. **CONTEXTUAL VIDEO ANALYSIS:**
   - Verify background consistency throughout the video
   - Check if the setting matches the claimed time and place
   - Look for anachronisms in clothing, technology, or environment
   - Examine if all elements in the scene are temporally consistent

5. **AUDIO-VISUAL VERIFICATION:**
   - Check if the voice matches the person's appearance and age
   - Verify if speech patterns are consistent with mouth movements
   - Notice any audio quality inconsistencies or artificial enhancements

Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'.

**IMPORTANT:** If you detect clear signs of deepfake or significant video manipulation that affects the claim's validity, lean towards 'incorrect' unless the statement explicitly acknowledges the manipulated nature."""
    
    def _get_video_verification_user_prompt(self, claim):
        """Get user prompt for video verification"""
        return f"""Please analyze the provided video with special attention to potential deepfakes, digital manipulations, or synthetic content, then evaluate the statement: "{claim}"

COMPREHENSIVE VIDEO ANALYSIS:
1. Examine facial features and expressions across all frames for deepfake indicators
2. Check for technical manipulation signs and editing artifacts
3. Verify motion consistency and natural behavior patterns
4. Analyze audio-visual synchronization and voice authenticity
5. Assess contextual and temporal consistency
6. Determine the relationship between the statement and the authentic video content

Provide your judgment: 'correct', 'incorrect', or 'unrelated'."""
    
    def _get_basic_verification_system_prompt(self):
        """Get basic system prompt for fallback verification"""
        return """You are a rigorous fact-checking assistant. Your task is to analyze a statement and a multimedia file, then judge whether the provided statement is 'correct', 'incorrect', or 'unrelated' to the file's content. 

Pay attention to potential digital manipulation or synthetic content that might affect the accuracy of your assessment.

Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'. Do not add any additional explanation or information."""
    
    def _get_basic_verification_user_prompt(self, claim):
        """Get basic user prompt for fallback verification"""
        return f"""Please evaluate the statement based on the provided multimedia file, considering potential manipulation: "{claim}"

Provide your judgment: 'correct', 'incorrect', or 'unrelated'."""
    
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
            
            # Generate response with parameters optimized for fact-checking
            generated_ids = self.model.generate(
                **inputs,
                use_audio_in_video=use_audio,
                return_audio=False,
                max_new_tokens=512,
                temperature=0.1,  # Lower temperature for more consistent fact-checking
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.processor.tokenizer.eos_token_id
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
    
    def analyze_deepfake_indicators(self, media_path, media_type='image'):
        """Specialized method for detailed deepfake analysis"""
        
        if media_type == 'image':
            system_prompt = """You are a specialized deepfake detection expert. Analyze the provided image for signs of digital manipulation, face swapping, or synthetic generation. Focus on technical indicators and artifacts."""
            
            analysis_prompt = """Perform a detailed deepfake analysis of this image:

1. **FACIAL ANALYSIS:**
   - Eye consistency (shape, color, reflections, shadows)
   - Mouth and teeth alignment and naturalness
   - Skin texture consistency across face
   - Facial boundary artifacts or blending issues
   - Symmetry and proportion analysis

2. **TECHNICAL INDICATORS:**
   - Compression artifacts around face vs. background
   - Color space inconsistencies
   - Lighting direction and shadow consistency
   - Edge detection around facial features
   - Pixel-level artifacts or smoothing

3. **CONTEXTUAL CLUES:**
   - Age consistency across facial features
   - Hair-face boundary naturalness
   - Expression-context matching

Provide a detailed technical analysis of potential manipulation indicators."""
            
        else:  # video
            system_prompt = """You are a specialized deepfake detection expert. Analyze the provided video for signs of digital manipulation, face swapping, temporal inconsistencies, or synthetic generation."""
            
            analysis_prompt = """Perform a detailed deepfake analysis of this video:

1. **TEMPORAL CONSISTENCY:**
   - Frame-to-frame facial feature stability
   - Lighting consistency across frames
   - Expression transition naturalness
   - Facial boundary stability

2. **AUDIO-VISUAL SYNC:**
   - Lip-sync accuracy
   - Voice-appearance matching
   - Speech pattern consistency

3. **MOTION ANALYSIS:**
   - Natural head movement patterns
   - Eye movement and blinking consistency
   - Micro-expression authenticity

4. **TECHNICAL ARTIFACTS:**
   - Compression inconsistencies
   - Frame rate variations
   - Color space anomalies
   - Temporal filtering artifacts

Provide a comprehensive technical analysis of potential deepfake indicators."""
        
        user_content = [{"type": "text", "text": analysis_prompt}]
        
        if media_type == 'image':
            user_content.append({"type": "image", "image": media_path})
        else:
            user_content.append({"type": "video", "video": media_path})
        
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
    
def _get_image_verification_system_prompt_standard(self):
    """Get standard system prompt for image verification (without deepfake detection)"""
    return """You are a rigorous fact-checking assistant. Your task is to analyze a statement and an image, then determine if the statement is 'correct', 'incorrect', or 'unrelated' to the image content.

ANALYSIS REQUIREMENTS:
1. Compare the statement with what you observe in the image
2. Check for factual accuracy of claims about people, places, objects, or events
3. Verify contextual information like time, location, or circumstances
4. Consider if the image provides sufficient evidence to support or refute the claim

Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'."""

def _get_image_verification_user_prompt_standard(self, claim):
    """Get standard user prompt for image verification"""
    return f"""Please analyze the provided image and evaluate the statement: "{claim}"

Provide your judgment: 'correct', 'incorrect', or 'unrelated'."""

def _get_video_verification_system_prompt_standard(self):
    """Get standard system prompt for video verification (without deepfake detection)"""
    return """You are a rigorous fact-checking assistant. Your task is to analyze a statement and a video, then determine if the statement is 'correct', 'incorrect', or 'unrelated' to the video content.

ANALYSIS REQUIREMENTS:
1. Compare the statement with what you observe in the video (both visual and audio)
2. Check for factual accuracy of claims about people, actions, events, or dialogue
3. Verify contextual information like time, location, or circumstances
4. Consider if the video provides sufficient evidence to support or refute the claim

Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'."""

def _get_video_verification_user_prompt_standard(self, claim):
    """Get standard user prompt for video verification"""
    return f"""Please analyze the provided video and evaluate the statement: "{claim}"

Provide your judgment: 'correct', 'incorrect', or 'unrelated'."""

def _get_basic_verification_system_prompt_standard(self):
    """Get standard basic system prompt for fallback verification"""
    return """You are a rigorous fact-checking assistant. Your task is to analyze a statement and a multimedia file, then judge whether the provided statement is 'correct', 'incorrect', or 'unrelated' to the file's content.

Your answer must be strictly limited to one of these three words: 'correct', 'incorrect', 'unrelated'."""

def _get_basic_verification_user_prompt_standard(self, claim):
    """Get standard basic user prompt for fallback verification"""
    return f"""Please evaluate the statement based on the provided multimedia file: "{claim}"

Provide your judgment: 'correct', 'incorrect', or 'unrelated'."""