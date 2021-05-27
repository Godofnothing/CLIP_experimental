import torch
import torchvision.transforms as T
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

from sklearn.metrics import confusion_matrix

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

inv_normalize = T.Normalize(
    mean=[-m_ / s_ for m_, s_ in zip(MEAN, STD)],
    std=[1 / s_ for s_ in STD]
)

class ClassificationVisualizer:

  def __init__(
      self,
      clip_wrapper,
      dataset,
      images_in_row=1
  ):
    self.clip_wrapper = clip_wrapper.cuda()
    self.dataset = dataset
    self.images_in_row = images_in_row

  def visualize_predictions(self, num_images):
    indices = np.random.randint(low=0, high=len(self.dataset), size=num_images)
    samples = [self.dataset[idx] for idx in indices]

    images = torch.stack([sample[0] for sample in samples])
    labels = [sample[1] for sample in samples]
    images_to_show = [
        np.clip(np.moveaxis(inv_normalize(image).numpy(), 0, 2), 0, 1) for image in images
    ]

    with torch.no_grad():
        pred_labels = self.clip_wrapper.classify(images)

    nrows, ncols = math.ceil(num_images / self.images_in_row), self.images_in_row

    fig, ax = plt.subplots(nrows, ncols, figsize = (6 * ncols, 6 * nrows))

    for idx, (image, true_label, pred_label) in enumerate(zip(images_to_show, labels, pred_labels)):
        ax[idx // ncols, idx % ncols].imshow(image)
        ax[idx // ncols, idx % ncols].set_title(
            f"{self.dataset.idx_to_class[pred_label.item()]} ({self.dataset.idx_to_class[true_label]})", 
            fontsize=20,
            color=("green" if true_label == pred_label else "red")
        )
        ax[idx // ncols, idx % ncols].axis('off')

    plt.tight_layout()

  def get_top_k_mistakes(self, loader, top_k=10):
      c = Counter([path.split('/')[-2] for path in self.dataset.paths_to_images])

      true_labels = []
      pred_labels = []

      for batch in loader:
        output = self.clip_wrapper.test_step([x_.cuda() for x_ in batch], 0)
        true_labels.extend(list(output['labels'].cpu().numpy()))
        pred_labels.extend(list(output['preds'].cpu().numpy()))

      conf_matrix = confusion_matrix(true_labels, pred_labels)
      offdiag_conf_matrix = conf_matrix - np.diag(np.diag(conf_matrix))

      top_idx = np.argsort(-offdiag_conf_matrix.flatten())[:top_k]
      top_val = -np.sort(-offdiag_conf_matrix.flatten())[:top_k]

      num_classes = len(self.dataset.idx_to_class)

      true_classes, pred_classes, iou = [], [], []
      for idx, val in zip(top_idx, top_val):
        true_class_idx, pred_class_idx = idx // num_classes, idx % num_classes
        true_class = self.dataset.idx_to_class[true_class_idx]
        pred_class = self.dataset.idx_to_class[pred_class_idx]
        class_iou = val / (c[true_class] + c[pred_class])
        true_classes.append(true_class)
        pred_classes.append(pred_class)
        iou.append(class_iou)

      return pd.DataFrame({"True class" : true_classes, "Pred class" : pred_classes, "Class IoU" : iou})