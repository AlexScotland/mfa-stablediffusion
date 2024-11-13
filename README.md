# MFA_StableDiffusion
Modular FastApi StableDiffusion Plugin, used to generate text to image.

This plugin can be used standalone, or other plugins can rely on it to utilize the pipelines.

# Routes
## /lora/
### GET
Returns a list of all LoRA's
```json
{
  "LoraName": {
    "name": "LoraName",
    "safetensor": "LoraName.safetensors",
    "model": "model",
    "keywords": [
      "a keyword",
      "that will",
      "trigger the lora"
    ],
    "directory": "the://folder/directory/to/the/LoraName.safetensors"
  }
}
```
### POST
Upload a Lora.  Since we are uploading a file, we use a FORM submission.

```
lora_file: Lora_file.safetensors
model: model_name
keywords: any keywords, that will, trigger, the, LoRA
```
## models
### GET
Returns a list of all model's
```json
[
  "stable_diffusion",
  "Flux-1"
]
```
## download
### PUT
Download a HuggingFace safetensor file from a model name.

Returns Downloaded Status on success.

Usage is via URL:
`PUT API_URL/download/?model_name=stable-diffusion-3.5-medium`

## Export
### POST
Extracts a downloaded `.safetensors` into a folder.

`POST API_URL/export/safetensor/?safetensor_name=table-diffusion-3.5-medium`

## Generate
### POST
Generates an image.

Returns the image.

```json
{
  "prompt": "Image to generate",
  "height": 1000,
  "width": 1000,
  "negative_prompt": "Things we want the image generator to avoid",
  "model": "model_to_use",
  "loras": [
    {
        "name": "LoraName",
        "weight": 1 // How aggressive we want the LoRA to weigh in, 0 - 1
    }
    ]
}
```