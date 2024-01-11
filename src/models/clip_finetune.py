import torch
from base_model import BaseModel
from transformers import CLIPModel, CLIPProcessor


class CustomCLIPModel(BaseModel):
    def __init__(self, model_name: str):
        super(CustomCLIPModel, self).__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        # Cuda
        self.processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)
        # Apple silicon
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        processed_images = self.processor(images=images, return_tensors="pt").to(
            self.device
        )
        return processed_images

    def process_text(self, texts: list[str]) -> torch.Tensor:
        processed_texts = self.processor(text=texts, return_tensors="pt").to(
            self.device
        )
        return processed_texts

    def forward(
        self, images: torch.Tensor, texts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_images, channels, height, width = images.shape
        reshaped_images = images.view(
            batch_size * num_images, channels, height, width
        ).to(self.device)

        processed_images = self.process_image(reshaped_images)
        image_embeddings = self.model.get_image_features(**processed_images)

        text_embeddings = torch.zeros(
            len(texts), image_embeddings.shape[-1], device=self.device
        )
        for idx, text in enumerate(texts):
            processed_phrase = self.process_text(text)
            text_embeddings[idx] = self.model.get_text_features(
                **processed_phrase
            ).squeeze(0)

        return text_embeddings, image_embeddings

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

    def save_checkpoint(self, filepath: str):
        torch.save(self.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location=self.device))
        print(f"Model loaded from {filepath}")
