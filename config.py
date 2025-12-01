
import torch

class Config:
    # Model settings
    MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
    MAX_LENGTH = 20
    NUM_BEAMS = 4
    TEMPERATURE = 0.9
    TOP_K = 50
    TOP_P = 0.95
    
    # Image settings
    MAX_IMAGE_SIZE = (512, 512)
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]
    
    # Performance settings
    CACHE_MODEL = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # UI settings
    DEFAULT_EXAMPLES = [
        "assets/examples/example1.jpg",
        "assets/examples/example2.jpg",
        "assets/examples/example3.jpg"
    ]