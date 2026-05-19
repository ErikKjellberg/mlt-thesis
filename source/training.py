from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import TrainingArguments, Trainer

from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.losses import SiameseDistanceMetric, ContrastiveLoss



class HierarchicalContrastiveLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        tensor_mapping,
        distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
        margin: float = 2/3,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(model, distance_metric, margin, size_average)
        self.distance_metric = distance_metric
        self.margin = margin
        self.model = model
        self.size_average = size_average
        self.tensor_mapping = tensor_mapping

    def get_config_dict(self) -> dict[str, Any]:
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = f"SiameseDistanceMetric.{name}"
                break

        return {"distance_metric": distance_metric_name, "margin": self.margin, "size_average": self.size_average}

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # map 0 -> [0,0], 1 -> [1,0] etc.
        labels = self.tensor_mapping.to(labels.device)[labels]
        coarse_labels = labels[:, 0]
        fine_labels = labels[:, 1]
        coarse_loss = self.contrastive_loss.forward(sentence_features, coarse_labels)
        fine_loss = self.contrastive_loss.forward(sentence_features, fine_labels)
        return coarse_loss + fine_loss


class WeightedMultiClassTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, train_sampler, dev_sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_sampler = train_sampler
        self.dev_sampler = dev_sampler

    # Override to use custom collate and sampling functions for multilabel classification
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(self.train_dataset, batch_sampler=self.train_sampler, collate_fn=self.data_collator)

    def get_dev_dataloader(self) -> DataLoader:
        if self.dev_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(self.dev_dataset, batch_sampler=self.dev_sampler, collate_fn=self.data_collator)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


class WeightedMultiLabelTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, train_sampler, dev_sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.train_sampler = train_sampler
        self.dev_sampler = dev_sampler

    # Override to use custom collate and sampling functions for multilabel classification
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(self.train_dataset, batch_sampler=self.train_sampler, collate_fn=self.data_collator)

    def get_dev_dataloader(self) -> DataLoader:
        if self.dev_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(self.dev_dataset, batch_sampler=self.dev_sampler, collate_fn=self.data_collator)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is None:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
