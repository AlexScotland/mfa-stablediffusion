import io
import os
from typing import List

from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image, StableDiffusionXLControlNetPipeline, StableVideoDiffusionPipeline
from fastapi import Response, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi import APIRouter

from ..helpers.pipeline import __clean_up_pipeline

from ..factories.lora_factory import LoRAFactory

from ..helpers.lora import get_all_loras_by_name
from ..helpers.directory import get_root_folder
from ..loras import create_all_loras
from ..models.abstract_image_pipeline import AbstractImagePipeline

# Input Objects
from ..serializers.base_image_request import BaseImageRequest
from ..serializers.image_with_lora_request import ImageWithLoras
from ..serializers.model_request import ModelRequest
from ..serializers.lora_request import LoraRequest

# Exporters
from ..export.export_yaml import YAMLExporter

MODEL_DIRECTORY = f"{get_root_folder()}/image_models/"

PREFIX = "/image_generation"

ROUTER = APIRouter(
    prefix = PREFIX,
    tags=["Text To Image Generation"]
)

@ROUTER.post("/lora/")
def upload_lora(
        lora_file: UploadFile = File(...),
        model: str = Form(None),
        keywords: List[str] = Form(None),
    ):
    """
        API endpoint for creating a new LoRA
    """
    if model not in get_all_models():
        return f"{model} not found, is it downloaded?"
    lora_file = LoRAFactory.create(
        lora_file = lora_file,
        model = model,
        keywords = keywords[0].split(",")
    )
    # Save the file into the LoRA directory
    try:
        lora_file.save('loras')
    except FileExistsError:
        return "lora path already exists"
    YAMLExporter().export(lora_file)
    create_all_loras()
    return lora_file.name

@ROUTER.get("/lora/")
def get_loras():
    return create_all_loras()

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
def generate_picture(image: ImageWithLoras):
    pipeline = AbstractImagePipeline(MODEL_DIRECTORY, image.model)
    image_store = io.BytesIO()
    for generated_image in pipeline.generate_image(
        image.prompt,
        height=image.height,
        width=image.width,
        negative_prompt=image.negative_prompt,
        loras = image.loras
        ):
        generated_image.save(image_store,"png")
        break

    # Cleanup our pipeline
    __clean_up_pipeline(pipeline)

    return Response(content=image_store.getvalue(), media_type="image/png")
