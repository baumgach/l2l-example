from functools import partial

import torch
import torch.nn as _nn
import torchvision.models as _models

from torchcross.models.lightning import SimpleCrossDomainClassifier


def resnet18_backbone(pretrained=False):
    weights = _models.ResNet18_Weights.DEFAULT if pretrained else None
    resnet = _models.resnet18(weights=weights, num_classes=1000)
    num_features = resnet.fc.in_features
    resnet.fc = _nn.Identity()
    return resnet, num_features


def main():
    hparams = {
        "lr": 1e-3,
    }

    # Create optimizer
    optimizer = partial(torch.optim.Adam, **hparams)

    # Create the lighting model with pre-trained resnet18 backbone
    model = SimpleCrossDomainClassifier(resnet18_backbone(pretrained=True), optimizer)

    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
    main()
