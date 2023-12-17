from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    An abstract base class for models for Visual-WSD dataset.

    Attributes:
        model: pretrained model.
        processor: wrapped model's image processor and tokenizer into a single processor.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.processor = None

    @abstractmethod
    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Process the images.

        Args:
            images (torch.Tensor): A tensor containing the one image or stacked multiple images.

        Returns:
            torch.Tensor: The processed images.
        """
        pass

    @abstractmethod
    def process_text(self, texts: list[str]) -> torch.Tensor:
        """
        Process the textual input.

        Args:
            texts (list[str]): textual content (descriptions of images)

        Returns:
            torch.Tensor: The processed text.
        """
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """
        The forward pass of the model. Should handle both text and image data, and return a tensor of logits,
        where on first place would be logit for target.

        Args:
            images (torch.Tensor): visual content
            texts (list[str]): textual content

        Returns:
            torch.Tensor: A tensor of logits of size [batch_size, 10].
        """
        pass
