import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import get_metrics


def train_one_epoch(model, data_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0.0

    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    for idx, batch in loop:
        texts = batch["context"]
        images = torch.cat(
            [batch["target"].unsqueeze(1), batch["candidate_images"]], dim=1
        )
        batch_size, num_images, _, _, _ = images.shape

        text_embeddings, image_embeddings = model(images, texts)
        subsamples = loss_fn.generate_subsamples(batch_size, num_images)
        loss = loss_fn(text_embeddings, image_embeddings, subsamples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    return average_loss


def validate_one_epoch(model, data_loader, loss_fn):
    model.eval()

    total_loss = 0.0
    correct_preds = []
    all_target_ranks = []

    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for idx, batch in loop:
            texts = batch["context"]
            images = torch.cat(
                [batch["target"].unsqueeze(1), batch["candidate_images"]], dim=1
            )
            batch_size, num_images, _, _, _ = images.shape

            # loss calulation
            text_embeddings, image_embeddings = model(images, texts)
            subsamples = loss_fn.generate_subsamples(batch_size, num_images)
            loss = loss_fn(text_embeddings, image_embeddings, subsamples)
            total_loss += loss.item()

            # evaluation for metrics
            logits = model.evaluate(images, texts)
            probs = F.softmax(logits, dim=1)

            top_prob, top_indices = torch.max(probs, dim=1)

            for i in range(len(top_indices)):
                correct_target = 1 if top_indices[i] == 0 else 0
                correct_preds.append(correct_target)

                rank = (probs[i].sort(descending=True)[1] == 0).nonzero(as_tuple=True)[
                    0
                ].item() + 1
                all_target_ranks.append(rank)

    metrics = get_metrics(correct_preds, all_target_ranks)

    average_loss = total_loss / len(data_loader)

    return average_loss, metrics


def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs):
    print(f"Model {type(model).__name__} started training, device: {model.device}\n")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)

        val_loss, metrics = validate_one_epoch(model, val_loader, loss_fn)

        print(
            f"==> Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        print(f"Metrics: {metrics}\n")

    print(f"Model {type(model).__name__} training complete\n")
