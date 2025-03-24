import yaml
from pathlib import Path

from ..models.lora import LoRA

def create_all_loras():
    lora_list = {}
    for folder in Path(__file__).parent.iterdir():
        if folder.is_dir() and not folder.name.endswith("__"):
            with open(f"{folder}/config.yml") as lora_config:
                loaded_lora = yaml.safe_load(lora_config)
                lora_object = LoRA(
                    name = loaded_lora['name'],
                    safetensor = loaded_lora['safetensor'],
                    model = loaded_lora['model'],
                    keywords = loaded_lora['keywords'],
                    directory = loaded_lora['directory']
                )
                lora_list[lora_object.name] = lora_object
    return lora_list