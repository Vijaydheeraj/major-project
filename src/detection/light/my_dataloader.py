import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):
    """
    Populates a list of training image file paths from the given directory.
    
    Args:
        lowlight_images_path (str): Path to the directory containing low-light images.
    
    Returns:
        List[str]: A shuffled list of image file paths.
    """
    image_list_lowlight = glob.glob(lowlight_images_path + "**/*.jpg", recursive=True)

    train_list = image_list_lowlight

    random.shuffle(train_list)

    return train_list


class lowlight_loader(data.Dataset):
    """
    A PyTorch Dataset for loading low-light images.
    """

    def __init__(self, lowlight_images_path):
        """
        Initializes the dataset with a list of low-light images.
        
        Args:
            lowlight_images_path (str): Path to the directory containing low-light images.
        """
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 256
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        """
        Retrieves an image from the dataset at the specified index.
        
        Args:
            index (int): Index of the image to retrieve.
        
        Returns:
            torch.Tensor: The processed image tensor of shape (3, 256, 256).
        """
        data_lowlight_path = self.data_list[index]
        data_lowlight = Image.open(data_lowlight_path, cv2.IMREAD_COLOR)
        data_lowlight = data_lowlight.resize((self.size, self.size), Image.ANTIALIAS)
        data_lowlight = (np.asarray(data_lowlight) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        """
        Returns the number of images in the dataset.
        
        Returns:
            int: Number of images.
        """
        return len(self.data_list)
