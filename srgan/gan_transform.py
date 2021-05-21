from PIL import Image
import torch
import torchvision.transforms as transforms
from models import GeneratorResNet
from torchvision.utils import make_grid
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

    def _define_generator(self, path_to_model):
        gen = GeneratorResNet()
        gen.load_state_dict(torch.load(path_to_model))
        return gen

    def __call__(self, sample):
        sample = self.lr_transform(sample)
        # sample = sample[None]
        logits = self.gen(sample)
        normalize = make_grid(logits, padding=0, normalize=True)
        batch = torch.split(normalize, split_size_or_sections=self.hr_size, dim=-1)

        return batch


