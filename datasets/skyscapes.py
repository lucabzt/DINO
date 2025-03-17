"""
This class loads the preprocessed skyscapes dataset. Please first use complete_processing.py to
crop the images and vectorize the ground truths from the original skyscapes dataset.
The folder structure should look like this:
SS_Multi_Lane
    train/val
        **ROOT_PATH** = processed
            images
            labels

The labels should be in json format and structured like this:
{
        'gt_classes': **List of class indices, e.g. [0,1,2,...,0]**
        'gt_boxes': **List of image coordinates (shape [num_instances, 4, 2])**
}
"""
import json
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import torch


class SkyScapes(Dataset):
    def __init__(self, root, tile_size=512):
        """
            root: root path of the dataset (see description above for details)
            tile_size: size of the images
        """
        self.root = root
        self.tile_size = tile_size
        images = sorted(filter(lambda i: i.endswith('.jpg'), os.listdir(os.path.join(self.root, 'images'))))
        self.image_paths = [os.path.join(root, 'images', i) for i in images]

        # Transformations
        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

    def __getitem__(self, index):
        im_path = self.image_paths[index]
        label_path = im_path.replace('images', 'labels').replace('.jpg', '.json')

        # Get image
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        if not (image.shape[1] == image.shape[2] == self.tile_size):
            raise Exception(f"Image of shape {image.shape} does not match tile size ({self.tile_size})")

        # Get label
        label = json.load(open(label_path, 'r'))
        label['gt_boxes'] = (torch.tensor(label['gt_boxes'], dtype=torch.float32) / self.tile_size)
        label['gt_classes'] = torch.tensor(label['gt_classes'], dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.image_paths)

# --- For testing only ---
if __name__ == '__main__':
    val = SkyScapes(root='/home/luca-bozzetti/Code/DLR/SkyScapes/ZIP/SS_Multi_Lane/val/processed')
    image_0, label_0 = val[0]
    import matplotlib.pyplot as plt
    from dataset.preprocess_skyscapes.visualize_vector import visualize_and_save
    visualize_and_save(image_0, label_0['gt_boxes'], 'TEST.png')
    plt.imshow(image_0)
    plt.show()
    print(label_0)
