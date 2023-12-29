from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbstractLoss(ABC, nn.Module):
    @abstractmethod
    def generate_subsamples(batch_size, num_images):
        """
        returns list of subsamples's embeddings indexes
        """
        pass

    @abstractmethod
    def forward(self, text_embeddings, image_embeddings, subsamples):
        """
        calculates loss
        """
        pass


class TripletLoss(AbstractLoss):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def generate_subsamples(self, batch_size, num_images):
        triplets = []
        for i in range(batch_size):
            anchor_idx = i
            positive_idx = i * num_images
            negative_indices = [i * num_images + j for j in range(num_images) if j != 0]
            for neg_idx in negative_indices:
                triplets.append((anchor_idx, positive_idx, neg_idx))
        return triplets

    def forward(self, text_embeddings, image_embeddings, subsamples):
        loss = torch.tensor(0.0, device=text_embeddings.device)
        for anchor_idx, positive_idx, negative_idx in subsamples:
            anchor = text_embeddings[anchor_idx]
            positive = image_embeddings[positive_idx]
            negative = image_embeddings[negative_idx]
            loss += F.triplet_margin_loss(
                anchor, positive, negative, margin=self.margin
            )
        return loss / len(subsamples)


class ContrastiveLoss(AbstractLoss):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def generate_subsamples(self, batch_size, num_images):
        pairs = []
        for i in range(batch_size):
            anchor_idx = i
            positive_idx = i * num_images
            negative_indices = [i * num_images + j for j in range(num_images) if j != 0]
            pairs.append((anchor_idx, positive_idx, 1))
            for neg_idx in negative_indices:
                pairs.append((anchor_idx, neg_idx, -1))
        return pairs

    def forward(self, text_embeddings, image_embeddings, subsamples):
        loss = torch.tensor(0.0, device=text_embeddings.device)
        for anchor_idx, other_idx, label in subsamples:
            anchor = text_embeddings[anchor_idx]
            other = image_embeddings[other_idx]
            label_tensor = torch.tensor([label], device=text_embeddings.device)
            loss += F.margin_ranking_loss(
                anchor, other, label_tensor, margin=self.margin
            )
        return loss / len(subsamples)
