import torch
from base_model import BaseModel
from transformers import AutoProcessor, GroupViTModel


class GroupVITModel(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/groupvit
    """

    def __init__(self, model_name):
        super().__init__()
        self.model = GroupViTModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name, do_rescale=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        processed_images = self.processor(
            images=images, return_tensors="pt", padding=True
        ).to(self.device)
        return processed_images

    def process_text(self, text: str) -> torch.Tensor:
        processed_texts = self.processor(
            text=[text], return_tensors="pt", padding=True
        ).to(self.device)
        return processed_texts

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        images = images.to(self.device)
        logits = torch.zeros(images.shape[0], images.shape[1])

        for batch_idx, sample_images in enumerate(images):
            processed_images = self.process_image(sample_images)
            processed_texts = self.process_text(texts[batch_idx])
            encoding = processed_images
            encoding.update(processed_texts)
            outputs = self.model(**encoding)
            logits[batch_idx] = outputs.logits_per_image.squeeze(1)

        return logits
