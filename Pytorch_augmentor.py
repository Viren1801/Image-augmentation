import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import Augmentor
import numpy as np


# Define the demo dataset
class PersonDataset(Dataset):
    """
    Sample dataset for Augmenter demonstration.
    The dataset will consist of just one sample image.
    """

    def __init__(self, image):
        self.image = image

    def __len__(self):
        return 1  # return 1 as we have only one image

    def __getitem__(self, idx):
        # Returns the augmented image

        # Initialize the pipeline
        p = Augmentor.DataPipeline([[np.array(image)]])

        # Apply augmentations
        p.random_brightness(probability=.5, min_factor=0.5, max_factor=1.75)  # good with annotations
        p.random_color(probability=.8, min_factor=0, max_factor=3)  # good with annotations
        p.random_contrast(probability=.8, min_factor=.8, max_factor=1.5)  # good with annotations
        p.random_erasing(probability=.6, rectangle_area=0.5)  # good with annotations
        p.random_distortion(probability=.4, grid_width=10, grid_height=10, magnitude=2)

        # Sample from augmentation pipeline
        images_aug = p.sample(100)

        # Get augmented image
        augmented_image = images_aug[0][0]

        # convert to tensor and return the result
        return TF.to_tensor(augmented_image)


# Initialize the dataset, pass the augmentation pipeline as an argument to init function
image_path = r"E:\augmenting"
train_ds = PersonDataset(image_path)

# Initialize the autoloader
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
print(train_loader)
