
import os
from PIL import Image
from torch.utils.data import Dataset

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
    self.idx_to_class = {idx : class_name for class_name, idx in self.class_to_idx.items()}

    paths_to_images = []
    for class_name in os.listdir(self.img_dir):
      for image_name in os.listdir(f'{self.img_dir}/{class_name}'):
        paths_to_images.append(f"{self.img_dir}/{class_name}/{image_name}")
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