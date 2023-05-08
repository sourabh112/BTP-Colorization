# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
import random
import tensorflow as tf

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder_input='A',subfolder_target='B', direction='AtoB', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path_input = os.path.join(image_dir, subfolder_input)
        self.input_path_target = os.path.join(image_dir, subfolder_target)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path_input))]
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path_input, self.image_filenames[index])
        input = Image.open(img_fn)
        
        img_fn = os.path.join(self.input_path_target, self.image_filenames[index])
        target = Image.open(img_fn)

        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
