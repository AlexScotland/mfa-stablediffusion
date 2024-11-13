from pydantic import BaseModel

class LoRAGenerationRequest(BaseModel):
    name: str
    weight: int
