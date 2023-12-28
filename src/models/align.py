import torch
from base_model import BaseModel
from transformers import AlignModel, AlignProcessor


class ALIGNModel(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/align
    """

    def __init__(self, model_name):
        super().__init__()
        self.model = AlignModel.from_pretrained(model_name)
        self.processor = AlignProcessor.from_pretrained(model_name)

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

    def evaluate(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        images = images.to(self.device)
        logits = torch.zeros(images.shape[0], images.shape[1], device=self.device)

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
#     model_name = "kakaobrain/align-base"
#     model = ALIGNMODEL(model_name=model_name)

#     images = torch.stack([load_image(url) for url in image_urls])
#     output = model(images, texts)
#     print(output)


# processor = AlignProcessor.from_pretrained("kakaobrain/align-base")
# model = AlignModel.from_pretrained("kakaobrain/align-base")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# candidate_labels = ["an image of a cat", "an image of a dog"]

# inputs = processor(text=candidate_labels, images=image, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# # this is the image-text similarity score
# logits_per_image = outputs.logits_per_image

# # we can take the softmax to get the label probabilities
# probs = logits_per_image.softmax(dim=1)
# print(probs)
