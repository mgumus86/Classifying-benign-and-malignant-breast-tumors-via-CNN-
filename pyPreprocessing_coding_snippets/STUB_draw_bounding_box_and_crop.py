import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import crop
import json
import math

# read input image
img = read_image(r'Data\Ultrasound-labeled\benign\1381668\20180115091728.jpg')

with open(r'Data\Ultrasound-labeled\benign\1381668\20180115091728.json') as json_file:
    label = json.load(json_file)
    
    position = label['shapes'][0]['points']

    print(position)

    y_min = math.floor(position[0][1])
    y_max = math.ceil(position[1][1])
    x_min = math.floor(position[0][0])
    x_max = math.ceil(position[1][0])

# bounding box in (xmin, ymin, xmax, ymax) format
# top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
bbox = [x_min, y_min, x_max, y_max]
bbox = torch.tensor(bbox, dtype=torch.int)
bbox = bbox.unsqueeze(0)

print(f"Original image dimensions: {img.size()}")

# draw bounding box on the input image
img_box=draw_bounding_boxes(img, bbox, width=3, colors=(255,0,0))

# transform it to PIL image and display
img_box = torchvision.transforms.ToPILImage()(img_box)
img_box.show()

croppedImg = crop(img, y_min, x_min, y_max-y_min, x_max-x_min)
print(f"Cropped image dimensions: {croppedImg.size()}")
croppedImg = torchvision.transforms.ToPILImage()(croppedImg)
croppedImg.show()