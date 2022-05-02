import json
import torch
import torchvision

from pycocotools.coco import COCO
from collections import defaultdict
import time
import csv
import os

from coco_utils import convert_coco_poly_to_mask, coco_remove_images_without_annotations
import transforms as T

class DRINKS(COCO):
    def __init__(self, csvFile=None, max_samples=None): # TODO: change max_samples to None

        self.dataset = {}
        self.anns = {}
        self.imgs = {}
        self.imgToAnns = defaultdict(list)
        self.catToAnns = defaultdict(list)

        self._max_samples = max_samples

        if csvFile:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = self._csvToCocoDataset(csvFile)
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
    
    def _csvToCocoDataset(self, csvFile):
        ds = {"images": [], "annotations": [], "categories": []}

        with open(csvFile, 'r') as f:
            lines = csv.reader(f)
            header = next(lines)

            img_temp = [] # list of image filenames

            ann_counter = 1
            for row in lines:
                frame, xmin, xmax, ymin, ymax, class_id = row

                image_id = int(frame.strip(".jpg"))

                ann = {}
                ann["id"] = ann_counter
                ann["image_file"] = frame       # TODO: Remove later
                ann["image_id"] = image_id
                ann["category_id"] = int(class_id)
                ann["bbox"] = [
                    int(xmin),
                    int(ymin),
                    int(xmax)-int(xmin),
                    int(ymax)-int(ymin),
                ]
                ann["segmentation"] = [[]]         # IGNORED
                ann["area"] = (int(xmax)-int(xmin))*(int(ymax)-int(ymin))
                ann["iscrowd"] = 0

                ds["annotations"].append(ann)
                ann_counter += 1
                
                if frame not in img_temp:
                    image = {}
                    image["id"] = image_id
                    image["file_name"] = frame
                    image["height"] = 480             # assumed all images in the csv are 480 x 640
                    image["width"] = 640

                    ds["images"].append(image)
                    img_temp.append(frame)
                
                if self._max_samples:
                    if len(img_temp) > self._max_samples:
                        break
            
            categories = [
                { "id": 1, "name": "Summit Drinking Water 500ml" },
                { "id": 2, "name": "Coca-Cola 330ml" },
                { "id": 3, "name": "Del Monte 100% Pineapple Juice 240ml" }
            ]
            ds["categories"] = categories

        # TODO: remove later
        # image_set = csvFile[7:].strip(".csv")
        # with open(f"instances_{image_set}", "w") as w:
        #     json.dump(ds, w) 

        return ds

class DrinksDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, csv_File, transforms, max_samples=None):
        super().__init__(img_folder, None)
        self._transforms = transforms

        self.coco = DRINKS(csv_File, max_samples)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class ConvertCocoPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        if keypoints is not None:
            keypoints = keypoints[keep]

        segmentations = [obj["segmentation"] for obj in anno]
        if segmentations[0][0]:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            masks = masks[keep]
        else:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        if masks is not None:
            target["masks"] = masks
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

def get_drinks(root, image_set, transforms, max_samples=None):
    csv_file = os.path.join(root, "labels_{}.csv".format(image_set))

    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = DrinksDetection(root, csv_file, transforms=transforms, max_samples=max_samples)

    if image_set == "train":
        dataset = coco_remove_images_without_annotations(dataset)
    
    return dataset


                