from models.qwen_model import QwenModel

class VideoProcessor:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B", cache_dir=None):
        self.model = QwenModel(model_path, cache_dir=cache_dir)
    
    def process(self, video_path):
        """
        Process a video and return its content description with separate visual and audio components
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary containing visual_content, audio_content, and full_content
        """
        video_content = self.model.process_video(video_path)
        return video_content