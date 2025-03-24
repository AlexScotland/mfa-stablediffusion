def create_all_loras():
    all_loras = []
    for lora_path in ALL_LORAS:
        all_loras.append(LoRAFactory.create(ALL_LORAS[lora_path], lora_path))
    return all_loras

def get_all_loras_by_name(list_of_name, all_loras):
    return_list = []
    for lora in list_of_names:
        return_list.append(all_loras.get(lora))
    return return_list