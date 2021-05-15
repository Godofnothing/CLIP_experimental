import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CLIP_Lite(pl.LightningModule):
  '''
  class that takes the pretrained backbone of clip model
  and can be then trained as in image classifier
  <classic> - trained as ordinary classifier with BCE loss
  <cosine_similarity> - model is trained to maximize cosine_similarity
  '''
  supported_training_modes = ['classic', 'cosine_similarity']

  def __init__(
      self,
      clip_model,
      training_mode='classic',
      clip_out_features=1024,
      num_classes=None,
  ):
    super().__init__()
    assert training_mode in self.supported_training_modes
    self.training_mode = training_mode
    self.model = clip_model
    #self.model = clip_model.to(dtype=torch.float32)

    # freeze the parameters of text_transformer to save memory
    for param in self.model.transformer.parameters():
      param.requires_grad = False

    if self.training_mode == 'classic':
      assert num_classes, "Number of classes has to be specified"
      self.classifier = nn.Linear(clip_out_features, num_classes).to(dtype=torch.float16)
      #self.classifier = nn.Linear(clip_out_features, num_classes).to(dtype=torch.float32)

  def forward(self, batch):
    image_batch, text_batch = batch

  def training_step(self, batch, batch_idx):
    image_batch, text_batch = batch
    
    image_features = self.model.visual(image_batch)

    if self.training_mode == 'classic':
      image_logits = self.classifier(image_features)
      image_labels = text_batch

      loss = F.cross_entropy(image_logits, image_labels)

    elif self.training_mode ==  'cosine_similarity':
      text_features = self.model.encode_text(text_batch)

      # normalize features
      image_features /= (image_features.norm(dim=-1, keepdim=True) + 1e-6)
      text_features /= (text_features.norm(dim=-1, keepdim=True) + 1e-6)

      # calculate loss
      loss = -torch.mean(image_features * text_features)

    self.log('train/loss', loss, on_step=True, on_epoch=True)

    return loss

  def validation_step(self, batch, batch_idx):
    image_batch, text_batch = batch
    
    image_features = self.model.visual(image_batch)

    if self.training_mode == 'classic':
      image_logits = self.classifier(image_features)
      image_labels = text_batch

      loss = F.cross_entropy(image_logits, image_labels)

    elif self.training_mode ==  'cosine_similarity':
      text_features = self.model.encode_text(text_batch)

      # normalize features
      image_features /= (image_features.norm(dim=-1, keepdim=True) + 1e-6)
      text_features /= (text_features.norm(dim=-1, keepdim=True) + 1e-6)

      # calculate loss
      loss = -torch.mean(image_features * text_features)

    self.log('val/loss', loss, on_step=True, on_epoch=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return {"optimizer" : optimizer, "scheduler" : scheduler, "monitor" : "val_loss"}

class CLIP_Pro(pl.LightningModule):
  '''
  class that performs training with positive and negative samples as in CLIP paper
  <visual> - only visual part is trained
  <visual&text> - both visual part and text transformer are trained
  '''
  supported_training_modes = ['visual', 'visual&text']

  def __init__(
      self,
      clip_model,
      training_mode='visual'
  ):
    super().__init__()
    assert training_mode in self.supported_training_modes
    self.training_mode = training_mode
    self.model = clip_model

    # freeze the parameters of text_transformer to save memory
    if self.training_mode == 'visual':
      for param in self.model.transformer.parameters():
        param.requires_grad = False

  def forward(self, batch):
    image_batch, text_batch = batch

    return self.model.visual(image_batch)

  def training_step(self, batch, batch_idx):
    image_batch, text_batch = batch

    image_features = self.model.visual(image_batch)
    text_features = self.model.encode_text(text_batch)

    # normalize features
    image_features /= (image_features.norm(dim=-1, keepdim=True) + 1e-6)
    text_features /= (text_features.norm(dim=-1, keepdim=True) + 1e-6)

    # calculate cosine similarities
    loss = torch.mean(image_features @ text_features.T) - 2 * torch.mean(image_features * text_features)

    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    return {"optimizer" : optimizer, "scheduler" : scheduler, "monitor" : "val_loss"}