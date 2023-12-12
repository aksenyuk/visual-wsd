
import torch
from PIL import Image
from torchvision import transforms
from transformers import BlipModel, BlipProcessor

import requests 
from io import BytesIO

from base_model import BaseModel


class BLIPModel(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/clip
    """
    def __init__(self, model_name):
        super().__init__()
        self.model = BlipModel.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)


    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        processed_images = self.processor(images=images, return_tensors="pt").to(self.device)
        return processed_images

    ## TODO - phrase / word?? 
    def process_text(self, texts: list) -> torch.Tensor:
        processed_texts = self.processor(text=texts, return_tensors="pt").to(self.device)
        return processed_texts


    def forward(self, images: torch.Tensor, texts: list):
        processed_images = self.process_image(images)
        processed_texts = self.process_text(texts)
        
        output = self.model(input_ids=processed_texts.input_ids, 
                            pixel_values=processed_images.pixel_values,
                            return_dict=True)
        ## similarity scores
        logits_per_image = output.logits_per_image  
        ## softmax for label probabilities
        probs = logits_per_image.softmax(dim=1)
        return logits_per_image, probs
    

## EVERYTHING BELOW IS CHECK 
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
]
texts = ["a photo of a cat", "a photo of a dog"]

def load_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    transform = transforms.ToTensor()
    return transform(img)


if __name__ == '__main__':
    model_name = 'Salesforce/blip-image-captioning-base'
    model = BLIPModel(model_name=model_name)

    images = torch.stack([load_image(url) for url in image_urls])
    output = model(images, texts)
    print(output)
