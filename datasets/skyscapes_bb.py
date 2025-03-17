import json
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import torch
from pathlib import Path


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
        h, w = image.shape[:2]
        image = self.transform(image)
        
        if not (image.shape[1] == image.shape[2] == self.tile_size):
            raise Exception(f"Image of shape {image.shape} does not match tile size ({self.tile_size})")
        
        # Get label
        label_data = json.load(open(label_path, 'r'))
        boxes_coords = torch.tensor(label_data['gt_boxes'], dtype=torch.float32)
        
        # Convert [num_instances, 4, 2] to [num_instances, 4] in format [x_center, y_center, width, height]
        boxes = []
        for box in boxes_coords:
            # Calculate min and max points
            min_x = torch.min(box[:, 0])
            min_y = torch.min(box[:, 1])
            max_x = torch.max(box[:, 0])
            max_y = torch.max(box[:, 1])
            
            # Convert to [cx, cy, w, h] format
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            w = max_x - min_x
            h = max_y - min_y
            
            # Normalize
            cx, w = cx / self.tile_size, w / self.tile_size
            cy, h = cy / self.tile_size, h / self.tile_size
            
            boxes.append([cx, cy, w, h])
        
        # Create target dictionary
        target = {}
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(label_data['gt_classes'], dtype=torch.int64)
        target["image_id"] = torch.tensor([index], dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([h, w])
        target["size"] = torch.as_tensor([h, w])
        
        return image, target

    def __len__(self):
        return len(self.image_paths)
    
def build(image_set, args):
    root = Path(args.coco_path)
    if image_set == "train":
        path = os.path.join(root, "train", "processed")
    else:
        path = os.path.join(root, "val", "processed")

    return SkyScapes(path, args.tile_size)
