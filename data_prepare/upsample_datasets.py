import glob
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch
import torch.nn.functional as f
import numpy as np
from torchvision.utils import save_image
from srgan.gan_transform import GANUpsample

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

class FruitsDatasetBilinear(Dataset):

    def __init__(self, images_path, path_to_save, new_shape=224):
        self.images = glob.glob(images_path + '/*')
        self.path_to_save = path_to_save

        splitted_path = images_path.split('/')

        self.phase = splitted_path[3]
        self.class_name = splitted_path[4]

        self.new_shape = new_shape
        self.to_tensor = ToTensor()

        self.new_directory = f'{self.path_to_save}/bilinear/{self.class_name}'
        if not os.path.exists(self.new_directory):
            os.makedirs(self.new_directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.to_tensor(np.asarray(image))

        batch = f.interpolate(image[None], size=self.new_shape, mode='bilinear', align_corners=True).squeeze(0)
        name = self.images[index].split('/')[-1]
        save_path = f'{self.new_directory}/{name}'

        return batch, save_path


class FruitsDatasetGAN(Dataset):

    def __init__(self, images_path, gan_path, path_to_save, new_shape=224):
        self.images = glob.glob(images_path + '/*')
        self.path_to_save = path_to_save

        splitted_path = images_path.split('/')

        self.phase = splitted_path[3]
        self.class_name = splitted_path[4]

        self.new_shape = new_shape
        self.gan_transform = GANUpsample(gan_path)

        self.new_directory = f'{self.path_to_save}/gan/{self.class_name}'
        if not os.path.exists(self.new_directory):
            os.makedirs(self.new_directory)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])

        with torch.no_grad():
            image = self.gan_transform(image)

            name = self.images[index].split('/')[-1]
            save_path = f'{self.new_directory}/{name}'

            return image, save_path


def phase_images(batch_size, phase_folders, path_to_save, mode, gan_path=None):
    print(f'Processing images...')

    for i in tqdm(range(len(phase_folders))):
        if mode == 'bilinear':
            fruit_dataset = FruitsDatasetBilinear(phase_folders[i], path_to_save)

        elif mode == 'gan':
            fruit_dataset = FruitsDatasetGAN(phase_folders[i], gan_path, path_to_save)

        else:
            raise NotImplementedError('Incorrect mode.')

        fruit_dataloader = DataLoader(fruit_dataset, num_workers=2, batch_size=batch_size, shuffle=False)
        for images, paths in fruit_dataloader:
            for image, path in zip(images, paths):
                save_image(image, path)


def image_upsampling(
        folder_path,
        path_to_save,
        batch_size=16,
        mode='bilinear',
        gan_path=None,
        phase='Training'
):
    assert phase in ['Traning', 'Test'], 'Incorrect phase'

    folders = glob.glob(os.path.join(folder_path, f'{phase}/*'))
    phase_images(batch_size, folders, path_to_save, mode, gan_path)


