import torch
from base_model import BaseModel
from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerProcessor


class BridgeTowerModel(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/bridgetower
    """

    def __init__(self, model_name):
        super().__init__()
        self.model = BridgeTowerForImageAndTextRetrieval.from_pretrained(model_name)
        self.processor = BridgeTowerProcessor.from_pretrained(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def process_image_text(self, image: torch.Tensor, text: str) -> torch.Tensor:
        encoding = self.processor(image, text, return_tensors="pt").to(self.device)
        return encoding

    def process_image(self):
        pass

    def process_text(self):
        pass

    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        images = images.to(self.device)
        logits = torch.zeros(images.shape[0], images.shape[1])

        for batch_idx, sample_images in enumerate(images):
            ## I couldn't make model process multiple images, so iterating one by one
            for candidate_idx, candidate_image in enumerate(sample_images):
                encoding = self.process_image_text(candidate_image, texts[batch_idx])
                outputs = self.model(**encoding)
                logits[batch_idx][candidate_idx] = outputs.logits[0, 1].item()

        return logits
