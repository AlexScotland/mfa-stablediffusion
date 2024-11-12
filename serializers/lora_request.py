from pydantic import BaseModel

class LoraRequest(BaseModel):
    model: str
    keywords: list
