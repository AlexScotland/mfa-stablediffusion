from typing import List

from .base_image_request import BaseImageRequest
from .lora_generation_request import LoRAGenerationRequest

class ImageWithLoras(BaseImageRequest):
    loras: List[LoRAGenerationRequest] = []