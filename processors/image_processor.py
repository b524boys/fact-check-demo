from models.qwen_model import QwenModel

class ImageProcessor:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B", cache_dir=None):
        self.model = QwenModel(model_path, cache_dir=cache_dir)
    
    def process(self, image_path):
        """
        Process an image and return its detailed content description
        
        Args:
            image_path: Path to the image file
            
        Returns:
            String containing the detailed image description
        """
        image_description = self.model.process_image(image_path)
        return image_description