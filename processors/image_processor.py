from models.qwen_model import QwenModel

class ImageProcessor:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B"):
        self.model = QwenModel(model_path)
    
    def process(self, image_path):
        image_text = self.model.process_image(image_path)
        return image_text