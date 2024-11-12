def create_all_loras():
    all_loras = []
    for lora_path in ALL_LORAS:
        all_loras.append(LoRAFactory.create(ALL_LORAS[lora_path], lora_path))
    return all_loras

def find_lora_by_name(name, model):
    for lora in LORAS:
        if lora == name and lora.base_model in model:
            return lora
    return None