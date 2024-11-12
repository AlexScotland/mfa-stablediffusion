from pydantic import BaseModel

class BaseImageRequest(BaseModel):
    prompt: str
    height: int = 600
    width: int = 600
    negative_prompt:str = ""
    model: str = None
