import os

from ..helpers.directory import get_root_folder

class LoRA():

    def __init__(
        self,
        name,
        safetensor,
        model,
        keywords,
        directory = None):
        self.name = name
        self.safetensor = safetensor
        self.model = model
        self.keywords = keywords
        self.directory = directory

    def save(self, directory):
        self.directory = f"{get_root_folder()}/{directory}/{self.safetensor.filename.replace('.safetensors', '')}"
        os.mkdir(self.directory)
        save_directory = f"{self.directory}/{self.safetensor.filename}"
        file_contents = self.safetensor.file.read()
        with open(save_directory, "wb") as safe_tensor_file:
            safe_tensor_file.write(file_contents)

    def to_dict(self):
        return {
            'name': self.name,
            'safetensor': self.safetensor.filename,
            'directory': self.directory,
            'keywords': self.keywords,
            'model': self.model
        }
