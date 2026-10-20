"""Training helpers for the Fashion-MNIST classification experiments."""

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, int]:
    """Train for one pass and return mean loss, accuracy, and example count.

    Batches contain images and labels, optionally followed by sample IDs.
    Images are already scaled and are passed to the model without rescaling.
    The supplied criterion must return a batch-mean loss, as the runner's
    default CrossEntropyLoss does. Accuracy is a fraction between zero and one.
    An empty dataloader raises ValueError because no mean can be measured.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        # Sample IDs are metadata and are not needed for training updates.
        images, labels = batch[:2]
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Weight batch means by their actual sizes, including partial batches.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("Cannot train on an empty dataloader.")

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        total_examples,
    )
