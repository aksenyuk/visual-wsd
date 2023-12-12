
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor

import requests 
from io import BytesIO

from base_model import BaseModel


class ClipModel(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/clip
    """
    def __init__(self, model_name):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = self.clip_model.to(self.device)


    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        processed_images = self.clip_processor(images=images, return_tensors="pt").to(self.device)
        return processed_images

    ## TODO - phrase / word?? 
    def process_text(self, texts: list) -> torch.Tensor:
        processed_texts = self.clip_processor(text=texts, return_tensors="pt").to(self.device)
        return processed_texts


    def forward(self, images: torch.Tensor, texts: list):
        processed_images = self.process_image(images)
        processed_texts = self.process_text(texts)
        
        output = self.clip_model(input_ids=processed_texts.input_ids, 
                                 pixel_values=processed_images.pixel_values,
                                 return_dict=True)
        return output
    

## EVERYTHING BELOW IS CHECK 
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]
texts = ["a photo of a cat", "a photo of a dog"]

def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return transforms.ToTensor(img)


if __name__ == '__main__':
    model_name = 'openai/clip-vit-base-patch32'
    clip_model = ClipModel(model_name=model_name)

    images = torch.stack([load_image(url) for url in image_urls])
    output = clip_model(images, texts)
    print(output)
    