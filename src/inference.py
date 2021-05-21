import torch
import torchvision.transforms as T
import numpy as np
import math
import matplotlib.pyplot as plt

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
            f"true : {self.dataset.idx_to_class[true_label]} \n pred : {self.dataset.idx_to_class[pred_label.item()]}", 
            fontsize=20
        )
        ax[idx // ncols, idx % ncols].axis('off')

    plt.tight_layout()
    