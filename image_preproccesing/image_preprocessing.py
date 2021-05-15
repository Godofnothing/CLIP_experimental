import glob
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch
import torch.nn.functional as f
import numpy as np
from torchvision.utils import save_image

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings('ignore')

from torchvision.utils import save_image

class FruitsDatasetBilinear(Dataset):
    
    def __init__(self, images_path, new_shape=224):

        self.images = glob.glob(images_path + '/*')
        splitted_path = images_path.split('/')
        self.phase = splitted_path[1]
        self.class_name = splitted_path[2]

        self.new_shape = new_shape
        self.to_tensor = ToTensor()

        self.new_directory = f'New{self.phase}/bilinear/{self.class_name}'
        if not os.path.exists(self.new_directory):
            os.makedirs(self.new_directory)
         
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.to_tensor(np.asarray(image))

        batch = f.interpolate(image[None], size=self.new_shape, mode='bilinear', align_corners=True).squeeze(0)
        save_path = self.images[index].replace('Fruit-Images-Dataset/', '')
        save_path = save_path.replace(self.phase, f'New{self.phase}/bilinear')

        return batch, save_path

def phase_images(batch_size, phase_folders, mode):

    print(f'Processing images...')

    for i in tqdm(range(len(phase_folders))):
        if mode == 'bilinear':
            fruit_dataset = FruitsDatasetBilinear(phase_folders[i])

        elif mode == 'gan':
            raise NotImplementedError('sad')

        else:
            raise ValueError('Incorrect mode!!!!!')

        fruit_dataloader = DataLoader(fruit_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

        for images, paths in fruit_dataloader:
            for image, path in zip(images, paths):
                save_image(image, path)

def image_upsampling(batch_size=32, mode='bilinear'):
    folder = 'Fruit-Images-Dataset'
    train_folders = glob.glob(os.path.join(folder, 'Training/*'))
    test_folders = glob.glob(os.path.join(folder, 'Test/*'))

    phase_images(batch_size, train_folders, mode)
    phase_images(batch_size, test_folders, mode)

    
    
