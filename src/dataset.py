import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler


class CLIPDataset(Dataset):

    def __init__(self, img_dir, image_transform=None, prompt_transform=None, return_indices=False):
        '''
        img_dir : str
          path to the directory with images
        image_transform : callable
          image augmentation
        prompt_transform : callable
          function generating prompts for class names
        return_indices : bool
          whether return indices or not
        '''
        self.img_dir = os.path.expanduser(img_dir)
        self.image_transform = image_transform
        self.prompt_transform = prompt_transform
        self.return_indices = return_indices

        self.class_to_idx = {}
        self.idx_to_class = {}
        self.paths_to_images = self._make_dataset()

    def _make_dataset(self):
        '''
        functions that iterates over the image folder and assigns to each image the class label
        '''
        if self.return_indices:
            for idx, class_name in enumerate(os.scandir(self.img_dir)):
                self.class_to_idx[class_name.name] = idx
        self.idx_to_class = {idx: class_name for class_name,
                             idx in self.class_to_idx.items()}

        paths_to_images = []
        for class_name in os.listdir(self.img_dir):
            for image_name in os.listdir(f'{self.img_dir}/{class_name}'):
                paths_to_images.append(
                    f"{self.img_dir}/{class_name}/{image_name}")
        return paths_to_images

    def __len__(self):
        return len(self.paths_to_images)

    def __getitem__(self, idx):
        '''
        return pair (image, prompt)
        '''
        img_path = self.paths_to_images[idx]

        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        prompt = img_path.split('/')[-2]

        if self.image_transform:
            image = self.image_transform(image)

        if self.prompt_transform:
            prompt = self.prompt_transform(prompt)

        if self.return_indices:
            prompt = self.class_to_idx[prompt]

        return image, prompt


class KPerClassSampler(Sampler):

    def __init__(self, dataset, k, seed):
        paths_to_images = dataset.paths_to_images
        grouped_by_class = self._group_by_class(paths_to_images)
        self.rng = np.random.default_rng(seed)
        self.sampled_images = self._sample_images(grouped_by_class, k)

    def _sample_images(self, grouped_by_class, k):
        sampled_img_paths = []
        for img_class, img_paths in grouped_by_class.items():
            if k > len(img_paths):
                replace = True
            else:
                replace = False

            sampled_img_paths.extend(
                self.rng.choice(img_paths, k, replace=replace)
                .tolist())
        return sampled_img_paths

    def _group_by_class(self, paths_to_images):
        grouped_by_class = defaultdict(list)
        for idx, img_path in enumerate(paths_to_images):
            # parse class name
            class_name = img_path.split('/')[-2]
            grouped_by_class[class_name].append((idx, img_path))
        return grouped_by_class

    def __len__(self):
        return len(self.sampled_images)

    def __iter__(self):
        self.rng.shuffle(self.sampled_images)
        return iter([int(idx) for idx, _ in self.sampled_images])


def train_val_split(dataset, val_size=0.1, random_state=42):
    train_idx, val_idx = train_test_split(
        np.arange(len(dataset)), test_size=val_size, random_state=random_state)
    train_dataset, val_dataset = deepcopy(dataset), deepcopy(dataset)

    train_dataset.paths_to_images = [
        dataset.paths_to_images[idx] for idx in train_idx]
    val_dataset.paths_to_images = [
        dataset.paths_to_images[idx] for idx in val_idx]

    return train_dataset, val_dataset
