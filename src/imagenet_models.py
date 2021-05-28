import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class ImageClassifier(pl.LightningModule):
    def __init__(self,
                 backbone,
                 backbone_out_features,
                 num_classes,
                 init_learning_rate=1e-4,
                 freeze_backbone=False):
        super(ImageClassifier, self).__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(backbone_out_features, num_classes)
        self.learning_rate = init_learning_rate

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('train/loss', loss)

        preds = torch.argmax(logits, 1)
        acc = torchmetrics.functional.accuracy(preds, labels)
        self.log('train/accuracy', acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True)

        preds = torch.argmax(logits, 1)
        acc = torchmetrics.functional.accuracy(preds, labels)
        self.log('val/accuracy', acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        preds = torch.argmax(logits, 1)
        return {'preds': preds, 'labels': labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([o['preds'] for o in outputs])
        labels = torch.cat([o['labels'] for o in outputs])
        acc = torchmetrics.functional.accuracy(preds, labels)

        self.log('test/accuracy', acc)

    def configure_optimizers(self):
        params = self.parameters() if not self.freeze_backbone else self.classifier.parameters()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        return optimizer