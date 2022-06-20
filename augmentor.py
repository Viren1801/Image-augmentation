# Import package
import Augmentor

# Initialize pipeline
p = Augmentor.Pipeline(r"E:\augmenting")


p.zoom_random(1, percentage_area=0.9)
p.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=8)
p.skew(probability=0.2)
p.RandomBrightness(probability=1, min_factor=1.5, max_factor=1.5)

# Sample from augmentation pipeline
images_aug = p.sample(100)

# Access augmented image and mask
print(images_aug)