from pydantic import BaseModel

class BaseImageRequest(BaseModel):
    prompt: str
    height: int = 600
    width: int = 600
    base_lora:str = None
    contextual_lora: str = None
    model: str = None
