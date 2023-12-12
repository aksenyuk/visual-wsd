from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    An abstract base class for models for Visual-WSD dataset.

    Attributes:
        image_processor: A nn.Module or similar object responsible for processing images.
        text_processor: A nn.Module or similar object responsible for processing text.
    """

    def __init__(self) -> None:
        super().__init__()
        self.image_processor = None
        self.text_processor = None

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
    def process_text(self, phrase: str, word: str) -> torch.Tensor:
        """
        Process the textual input.

        Args:
            phrase (str): The phrase containing the ambiguous word.
            word (str): The ambiguous word itself.

        Returns:
            torch.Tensor: The processed text.
        """
        pass

    @abstractmethod
    def forward(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        The forward pass of the model. Should handle both text and image data, and return a tensor of logits,
        where on first place would be logit for target.

        Args:
            data (Dict[str, torch.Tensor]): A dictionary containing textual and visual inputs. Expected keys are 'word', 'context', 'target', and 'other_images'.

        Returns:
            torch.Tensor: A tensor of logits of size [batch_size, 10].
        """
        pass
