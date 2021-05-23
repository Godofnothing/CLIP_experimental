import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class CLIP_Lite(pl.LightningModule):
  '''
  class that takes the pretrained backbone of clip model
  and can be then trained as in image classifier
  it is trained as an ordinary classifier with CE loss
  '''

  def __init__(
      self,
      clip_model,
      clip_out_features=1024,
      freeze_visual=False,
      num_classes=None,
      init_learning_rate=1e-4
  ):
    super().__init__()
    self.model = clip_model.to(dtype=torch.float32)
    self.learning_rate = init_learning_rate

    # freeze the image encoder, if needed
    if freeze_visual:
      for param in self.model.visual.parameters():
        param.requires_grad = False

    # freeze the parameters of text_transformer to save memory
    for param in self.model.transformer.parameters():
      param.requires_grad = False

    assert num_classes, "Number of classes has to be specified"
    self.classifier = nn.Linear(clip_out_features, num_classes).to(dtype=torch.float32)

  def forward(self, batch):
    image_batch, _ = batch
    
    image_features = self.model.visual(image_batch)

    return self.classifier(image_features)

  def training_step(self, batch, batch_idx):
    image_batch, true_labels = batch

    image_features = self.model.visual(image_batch)
    
    image_logits = self.classifier(image_features)

    loss = F.cross_entropy(image_logits, true_labels)
    self.log('train/loss', loss)

    pred_labels = image_logits.argmax(dim=-1)
    acc = torchmetrics.functional.accuracy(pred_labels, true_labels)
    self.log('train/accuracy', acc, on_step=True, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    image_batch, true_labels = batch

    image_features = self.model.visual(image_batch)

    image_logits = self.classifier(image_features)

    loss = F.cross_entropy(image_logits, true_labels)
    self.log('val/loss', loss)

    pred_labels = image_logits.argmax(dim=-1)
    acc = torchmetrics.functional.accuracy(pred_labels, true_labels)
    self.log('val/accuracy', acc)

  def test_step(self, batch, batch_idx):
    image_batch, true_labels = batch

    image_features = self.model.visual(image_batch)

    image_logits = self.classifier(image_features)
    pred_labels = image_logits.argmax(dim=-1)
    return {'preds': pred_labels, 'labels': true_labels}

  def test_epoch_end(self, outputs):
    preds = torch.cat([o['preds'] for o in outputs])
    labels = torch.cat([o['labels'] for o in outputs])
    acc = torchmetrics.functional.accuracy(preds, labels)

    self.log('test/accuracy', acc)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    return {"optimizer" : optimizer, "scheduler" : scheduler, "monitor" : "val/accuracy"}

  def classify(self, image_batch):
    image_batch = image_batch.cuda()
    image_features = self.model.visual(image_batch)
    return self.classifier(image_features).argmax(axis=-1)


class CLIP_Pro(pl.LightningModule):
  '''
  class that performs training with captions
  '''

  def __init__(
      self,
      clip_model,
      tokenized_captions,
      clip_out_features=1024,
      training_mode='visual',
      init_T=0.07,
      init_learning_rate=1e-4,
  ):
    super().__init__()

    self.model = clip_model.to(dtype=torch.float32)

    num_classes, captions_per_class, _ = tokenized_captions.shape

    self.T = nn.Parameter(torch.tensor(init_T, dtype=torch.float32, requires_grad=True, device='cuda'))
    self.caption_weights = nn.Parameter(
      torch.tensor([1 / captions_per_class for _ in range(captions_per_class)], 
      dtype=torch.float32, requires_grad=True, device='cuda'),
    )[None, :, None]

    self.learning_rate = init_learning_rate

    # freeze the parameters of text_transformer to save memory
    for param in self.model.transformer.parameters():
      param.requires_grad = False

    # initialize tensor with all captions
    self.caption_embeddings = torch.zeros(
      (num_classes, captions_per_class, clip_out_features), 
      requires_grad=False,
      device='cuda'
    )

    for class_idx, class_captions in enumerate(tokenized_captions):
      class_features = self.model.encode_text(class_captions)
      class_features = class_features / (class_features.norm(dim=-1, keepdim=True) + 1e-6)
      self.caption_embeddings[class_idx] = class_features.detach()

  def forward(self, batch):
    image_batch, true_labels = batch

    return self.model.visual(image_batch)

  def training_step(self, batch, batch_idx):
    image_batch, true_labels = batch

    image_features = self.model.visual(image_batch)

    # normalize features
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    class_features = torch.sum(self.caption_embeddings * self.caption_weights, axis=1)

    image_logits = torch.exp(self.T) * image_features @ class_features.T
    loss = F.cross_entropy(image_logits, true_labels)
    self.log('train/loss', loss)

    pred_labels = image_logits.argmax(dim=-1)
    acc = torchmetrics.functional.accuracy(pred_labels, true_labels)
    self.log('train/accuracy', acc, on_step=True, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    image_batch, true_labels = batch

    image_features = self.model.visual(image_batch)

    # normalize features
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    class_features = torch.sum(self.caption_embeddings * self.caption_weights, axis=1)

    image_logits = torch.exp(self.T) * image_features @ class_features.T
    loss = F.cross_entropy(image_logits, true_labels)
    self.log('val/loss', loss)

    pred_labels = image_logits.argmax(dim=-1)
    acc = torchmetrics.functional.accuracy(pred_labels, true_labels)
    self.log('val/accuracy', acc)

  def test_step(self, batch, batch_idx):
    image_batch, true_labels = batch

    image_features = self.model.visual(image_batch)

    # normalize features
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    class_features = torch.sum(self.caption_embeddings * self.caption_weights, axis=1)

    image_logits = torch.exp(self.T) * image_features @ class_features.T
    pred_labels = image_logits.argmax(dim=-1)
    return {'preds': pred_labels, 'labels': true_labels}

  def test_epoch_end(self, outputs):
    preds = torch.cat([o['preds'] for o in outputs])
    labels = torch.cat([o['labels'] for o in outputs])
    acc = torchmetrics.functional.accuracy(preds, labels)

    self.log('test/accuracy', acc)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
    return {"optimizer" : optimizer, "scheduler" : scheduler, "monitor" : "val/loss"}

  def classify(self, image_batch):
    image_batch = image_batch.cuda()
    image_features = self.model.visual(image_batch)
    class_features = torch.sum(self.caption_embeddings * self.caption_weights, axis=1)
    image_logits = torch.exp(self.T) * image_features @ class_features.T

    return image_logits.argmax(dim=-1)

