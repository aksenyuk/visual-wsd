import torch
from base_model import BaseModel
from transformers import BlipModel, BlipProcessor


class BLIPModel(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/blip
    """

    def __init__(self, model_name):
        super().__init__()
        self.model = BlipModel.from_pretrained(model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        processed_images = self.processor(images=images, return_tensors="pt").to(
            self.device
        )
        return processed_images

    def process_text(self, text: str) -> torch.Tensor:
        processed_texts = self.processor(text=text, return_tensors="pt").to(self.device)
        return processed_texts

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        images = images.to(self.device)
        logits = torch.zeros(images.shape[0], images.shape[1])

        for idx, sample_images in enumerate(images):
            processed_sample_images = self.process_image(sample_images)
            processed_phrase = self.process_text(texts[idx])

            output = self.model(
                input_ids=processed_phrase.input_ids,
                pixel_values=processed_sample_images.pixel_values,
                return_dict=True,
            )
            logits[idx] = output.logits_per_image.squeeze(1)

        return logits


## EVERYTHING BELOW IS CHECK

# from io import BytesIO
# import requests
# from PIL import Image
# from torchvision import transforms

# image_urls = [
#     "http://images.cocodataset.org/val2017/000000039769.jpg",
# ]
# texts = ["a photo of a cat", "a photo of a dog"]


# def load_image(url):
#     response = requests.get(url)
#     img = Image.open(BytesIO(response.content)).convert("RGB")
#     transform = transforms.ToTensor()
#     return transform(img)


# if __name__ == "__main__":
#     model_name = "Salesforce/blip-image-captioning-base"
#     model = BLIPMODEL(model_name=model_name)

#     images = torch.stack([load_image(url) for url in image_urls])
#     output = model(images, texts)
#     print(output)
