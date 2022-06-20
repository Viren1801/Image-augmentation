import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

# Define the demo dataset
class DogDataset3(Dataset):
    '''
    Sample dataset for Augmentor demonstration.
    The dataset will consist of just one sample image.
    '''

    def __init__(self, image):
        self.image = image

    def __len__(self):
        return 1 # return 1 as we have only one image

    def __getitem__(self, idx):
        # Returns the augmented image
        
        # Initialize the pipeline
        p = Augmentor.DataPipeline([[np.array(image)]])

        # Apply augmentations
        p.rotate(0.5, max_left_rotation=10, max_right_rotation=10) # rotate the image with 50% probability
        p.shear(0.5, max_shear_left = 10, max_shear_right = 10) # shear the image with 50% probability
        p.zoom_random(0.5, percentage_area=0.7) # zoom randomly with 50% probability

        # Sample from augmentation pipeline
        images_aug = p.sample(1)
        
        # Get augmented image
        augmented_image = images_aug[0][0]
        
        # convert to tensor and return the result
        return TF.to_tensor(augmented_image)

# Initialize the dataset, pass the augmentation pipeline as an argument to init function
train_ds = DogDataset3(image)

# Initialize the dataloader
trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)