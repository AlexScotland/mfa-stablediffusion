import torch
from diffusers import StableDiffusionPipeline

from .image_pipeline import ImagePipeline
from ..loras import create_all_loras

class AbstractImagePipeline(ImagePipeline):

    def generate_image(
            self,
            prompt,
            height = 600,
            width = 600,
            negative_prompt = "easynegative, human, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,",
            steps = 50,
            number_of_images = 1,
            loras = []):
        try:
            image = self.create_image(
                prompt,
                height,
                width,
                negative_prompt,
                steps,
                number_of_images,
                loras
            )
        except AttributeError:
            # Raised when we call functions only existing in SDXL, try SD 1.5 compatibility
            self.pipeline = self._generate_pipeline(StableDiffusionPipeline)
            image = self.create_image(
                prompt,
                height,
                width,
                negative_prompt,
                steps,
                number_of_images,
                loras
            )
        return image
        
    def create_image(
            self,
            prompt,
            height,
            width,
            negative_prompt,
            steps,
            number_of_images,
            loras = []
        ):
            for lora in loras:
                current_lora = create_all_loras().get(lora.name)
                current_lora.weight = lora.weight
                self.pipeline.load_lora_weights(
                            f"{current_lora.directory}/{current_lora.safetensor}",
                            weight_name=current_lora.safetensor,
                            adapter_name="contextual"
                            )
                self.pipeline.set_adapters(["contextual"], adapter_weights=[current_lora.weight])
                self.pipeline.fuse_lora(adapter_names=["contextual"], lora_scale=1.0)
            self.pipeline.unload_lora_weights()
            generated_images = self.pipeline(
                                    prompt,
                                    height=height,
                                    width=width,
                                    negative_prompt=negative_prompt,
                                    num_images_per_prompt=number_of_images
                                    ).images
            # Teardown
            if len(loras) > 0:
                self.pipeline.unfuse_lora()
                self.pipeline.unload_lora_weights()
            return generated_images