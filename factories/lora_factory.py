from ..models.lora import LoRA

class LoRAFactory():

    @staticmethod
    def create(
            lora_file,
            model,
            keywords
            ):
        return LoRA(
            name = lora_file.filename.replace(".safetensors",""),
            safetensor = lora_file,
            model = model,
            keywords = keywords
        )
