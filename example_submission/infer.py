import pickle
from functools import partial

import lightning.pytorch as pl
import torch
import torch.nn as _nn
import torchcross as tx
import torchvision.models as _models
from torch.utils.data import BatchSampler, RandomSampler, DataLoader
from torchcross.data import TaskDescription
from torchcross.models.lightning import SimpleCrossDomainClassifier


def resnet18_backbone(pretrained=False):
    weights = _models.ResNet18_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet18(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def main(args):
    task_path = args.task_path
    result_path = args.result_path

    batch_size = 64
    num_workers = 4

    hparams = {
        "lr": 1e-3,
    }

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    # Create the lighting model with pre-trained resnet18 backbone
    model = SimpleCrossDomainClassifier(resnet18_backbone(pretrained=False), optimizer)
    model.load_state_dict(torch.load("model.pt"))

    with open(task_path, "rb") as f:
        task = pickle.load(f)

    task_description = TaskDescription(task.task_target, task.classes, "target")

    # support_datapoints = list(zip(*task.support))
    support_datapoints = task.support

    sampler = RandomSampler(support_datapoints)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    collate_fn = tx.utils.collate_fn.stack

    batched_support_datapoints = [
        (collate_fn([support_datapoints[i] for i in batch]), task_description)
        for batch in batch_sampler
    ]

    target_train_dataloader = DataLoader(
        batched_support_datapoints,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=tx.utils.collate_fn.identity,
        pin_memory=True,
    )

    target_trainer = pl.Trainer(inference_mode=False, max_epochs=100)

    # Fine-tune the model on the task`s support set
    target_trainer.fit(model, target_train_dataloader, target_train_dataloader)

    # Get predictions on the query set
    query_datapoints = tx.utils.collate_fn.stack(task.query)
    preds = model(query_datapoints[0], task_description)

    with open(result_path, "wb") as f:
        pickle.dump(preds, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task_path", type=str)
    parser.add_argument("result_path", type=str)
    args = parser.parse_args()

    main(args)
