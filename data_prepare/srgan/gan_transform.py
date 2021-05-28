from PIL import Image
import torch
import torchvision.transforms as transforms
from models import GeneratorResNet
import numpy as np

class GANUpsample:
    """
    Photo-Realistic Single Image Super-Resolution
    Using a Generative Adversarial Network

    Args:
        path_to_model (str): root path to generator model
    """

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, path_to_model):
        self.gen = self._define_generator(path_to_model)
        self.gen.eval()

        self.hr_size = 224
        self.lr_transform = self._define_transform()

    def _define_transform(self):
        lr_transform = transforms.Compose(
            [
                transforms.Resize((self.hr_size // 4, self.hr_size // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(self.MEAN, self.STD),
            ]
        )

        return lr_transform

    def norm_ip(self, img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))
        return img

    def norm_range(self, t):
        return self.norm_ip(t, float(t.min()), float(t.max()))

    def _define_generator(self, path_to_model):
        gen = GeneratorResNet()
        gen.load_state_dict(torch.load(path_to_model))
        return gen

    def __call__(self, sample):
        sample = self.lr_transform(sample)
        logits = self.gen(sample[None])
        logits = self.norm_range(logits).squeeze(0)

        return logits


