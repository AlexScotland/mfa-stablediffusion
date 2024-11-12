import io
import os

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, StableVideoDiffusionPipeline
from fastapi import Response, UploadFile, Depends, File
from fastapi.responses import FileResponse
from fastapi import APIRouter

from ..helpers.pipeline import __clean_up_pipeline

# from factories.lora_factory import LoRAFactory
# from models.LoRA.lora_conf import ALL_LORAS
from ..helpers.lora import find_lora_by_name
from ..helpers.directory import get_root_folder
# from ..loras import LORAS
from ..models.abstract_image_pipeline import AbstractImagePipeline
from ..serializers.base_image_request import BaseImageRequest
from ..serializers.model_request import ModelRequest

MODEL_DIRECTORY = f"{get_root_folder()}/image_models/"

PREFIX = "/image_generation"

ROUTER = APIRouter(
    prefix = PREFIX,
    tags=["Text To Image Generation"]
)

@ROUTER.post("/loras/")
def all_lora_full_details(image: ModelRequest):
    ret_list = []
    for lora in LORAS:
        if lora.base_model in image.model:
            ret_list.append(lora)
    return ret_list

@ROUTER.get("/models/")
def get_all_models():
    model_dir = MODEL_DIRECTORY
    if not model_dir.endswith("/"): model_dir += "/"
    return [model_file for model_file in os.listdir(model_dir) if os.path.isdir(model_dir+model_file)]


@ROUTER.put("/download/")
def download_model_from_hugging_face(model_name: str):
    pipeline = DiffusionPipeline.from_pretrained(model_name)
    pipeline.save_pretrained(f"{MODEL_DIRECTORY}/{model_name.split('/')[-1]}")

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)
    return {"Status": "Downloaded"}

@ROUTER.post("/export/safetensor/")
def export_safetensor_local(safetensor_name: str):
    # TODO: Move this to another export directory,
    # have url and filetype be same variable
    extension = ".safetensors"
    if not safetensor_name.endswith(extension):
        safetensor_name+=extension
    
    safetensor_directory = f"{MODEL_DIRECTORY}/{safetensor_name}"
    model_export_directory = safetensor_directory.replace(extension, "")
    try:
        pipeline = StableDiffusionXLPipeline.from_single_file(
        safetensor_directory,
        local_files_only=True,
        use_safetensors=True
        )
    except TypeError as type_error_message:
        if "tokenizer" or "encoder" in type_error_message:
            pipeline = StableDiffusionPipeline.from_single_file(
                safetensor_directory,
                local_files_only=True,
                use_safetensors=True
                )
        else:
            raise type_error_message
    pipeline.save_pretrained(model_export_directory)

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)

    # Remove the old safe tensor
    os.remove(safetensor_directory)
    return {"model": safetensor_name.replace(extension, "")}

@ROUTER.post("/generate/")
def generate_picture(image: BaseImageRequest):
    # TODO: Dreambooth instead of base_lora
    # base_lora = find_lora_by_name(
    #     image.base_lora,
    #     image.model
    #     )
    # contextual_lora = find_lora_by_name(
    #     image.contextual_lora,
    #     image.model
    #     )
    pipeline = AbstractImagePipeline(MODEL_DIRECTORY, image.model, base_lora=None)
    image_store = io.BytesIO()
    for generated_image in pipeline.generate_image(
        image.prompt,
        height=image.height,
        width=image.width,
        negative_prompt=image.negative_prompt
        ):
        generated_image.save(image_store,"png")
        break

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)

    return Response(content=image_store.getvalue(), media_type="image/png")
