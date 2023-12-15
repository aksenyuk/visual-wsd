import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_metrics


def evaluate_model(
    model: torch.nn.Module, data_loader: DataLoader
) -> dict[str, float | list]:
    start_time = time.time()
    model.eval()

    predicted_images = []  # store which image was predicted
    correct_preds = []  # store whether the target was correctly predicted (1) or (0)
    all_target_ranks = []  # store the rank of the target in each prediction
    phrases = []  # store input phrases for further analysis
    all_probs = []  # store the probabilities for further analysis

    loop = tqdm(enumerate(data_loader), total=len(data_loader))

    with torch.no_grad():
        for idx, batch in loop:
            phrases.extend(list(batch["context"]))
            texts = batch["context"]

            target, candidate_images = batch["target"], batch["candidate_images"]
            images = torch.cat([target.unsqueeze(1), candidate_images], dim=1)

            logits = model(images, texts)
            probs = F.softmax(logits, dim=1)

            top_prob, top_indices = torch.max(probs, dim=1)
            predicted_images.extend([pred.item() for pred in top_indices])

            for i in range(len(top_indices)):
                correct_target = 1 if top_indices[i] == 0 else 0
                correct_preds.append(correct_target)

                rank = (probs[i].sort(descending=True)[1] == 0).nonzero(as_tuple=True)[
                    0
                ].item() + 1
                all_target_ranks.append(rank)

                all_probs.append(probs[i].tolist())

    accuracy, f1, precision, recall, mrr = get_metrics(correct_preds, all_target_ranks)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "time": time.time() - start_time,
        "phrases": phrases,
        "predictions": predicted_images,
    }
