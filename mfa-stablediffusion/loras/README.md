# LoRA's
## Structure
LoRA's can be added via the API.

When uploading a LoRA, you must provide the following:
1. LoRA `.safetensors` file
2. Any keywords associated with the LoRA
3. The base model that the LoRA pertains to.

When uploaded, the API:
1. Creates a folder named the safetensors name
2. Stores the safetensor
3. Creates a YAML file representing the object storing:
   1. keywords
   2. basemodel