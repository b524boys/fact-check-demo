from models.qwen_model import QwenModel

class VideoProcessor:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B"):
        self.model = QwenModel(model_path)
    
    def process(self, video_path):
        video_text = self.model.process_video(video_path)
        return video_text