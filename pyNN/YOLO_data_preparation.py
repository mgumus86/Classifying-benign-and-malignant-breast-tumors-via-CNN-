import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import config
import glob
import cv2
import math
import json
from pathlib import Path

random.seed(18)
# Link to skeleton file: https://blog.paperspace.com/train-yolov5-custom-data/

# USER SETTING - DIFFERENTIATE BETWEEN BENIGN AND MALIGNANT? THIS IS IMPORTANT FOR PREPPING THE YOLO DATASET
differentiate_option = True

if differentiate_option:
    class_name_to_id_mapping = {'benign':0,
                                'malignant':1}
else:
    class_name_to_id_mapping = {'tumor':0}

def extract_bounding_box(json_file, tumorClass, filename, differentiate=differentiate_option):
    # json_file = JSON file handle
    # tumorClass = 'benign' or 'malignant'
    # filename = Filename (with .jpg extension)
    # differentiate = log the class of the tumor or just label as a generic tumor
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    label = json.load(json_file)
    position = label['shapes'][0]['points']
    y_min = math.floor(position[0][1])
    y_max = math.ceil(position[1][1])
    x_min = math.floor(position[0][0])
    x_max = math.ceil(position[1][0])

    # Put in the filename
    info_dict['filename'] = filename
    info_dict['image_size'] = (label['imageWidth'], label['imageHeight'], 3)
    bbox = {}
    if differentiate:
        bbox['class'] = tumorClass
    else:
        bbox['class'] = 'tumor'

    bbox['xmin'] = x_min
    bbox['xmax'] = x_max
    bbox['ymin'] = y_min
    bbox['ymax'] = y_max
    info_dict['bboxes'].append(bbox)

    return info_dict



def convert_to_yolov5(info_dict, annotation_path):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save 
    save_file_name = os.path.join(annotation_path, info_dict["filename"].replace("jpg", "txt"))
    
    # Save the annotation to disk 
    print("\n".join(print_buffer), file= open(save_file_name, "w"))

# Write a function to get the data from JSON files
def prep_yolov5_data(DATA_DIR, 
                    benign_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "benign"), 
                    malig_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "malignant"),
                    differentiate=differentiate_option):
    
    # Check if the images folder has been created or not
    image_path = os.path.join(DATA_DIR, 'images')
    annotation_path = os.path.join(DATA_DIR, 'annotations')
    if not os.path.isdir(image_path):
        Path(image_path).mkdir(parents=True, exist_ok=True)
    # Check if annotations folder has been created or not
    if not os.path.isdir(annotation_path):
        Path(annotation_path).mkdir(parents=True, exist_ok=True)

    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    types = set()

    # Loop through the Data folder and pull the benign photos into the benign folder
    for rootfile, dirs, files in os.walk(benign_folder):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                # Read in accompanying JSON file here and convert to YOLO format
                jsonPath = file[:-4]+'.json'
                with open(os.path.join(rootfile,jsonPath)) as json_file:
                    this_info = extract_bounding_box(json_file, 'benign', file, differentiate)
                    convert_to_yolov5(this_info, annotation_path)

                # Copy the file to the new folder
                shutil.copy2(os.path.join(rootfile,file), image_path)

    # Do the same for the malignant folder
    for rootfile, dirs, files in os.walk(malig_folder):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                # Read in accompanying JSON file here and convert to YOLO format
                jsonPath = file[:-4]+'.json'
                with open(os.path.join(rootfile,jsonPath)) as json_file:
                    this_info = extract_bounding_box(json_file, 'malignant', file, differentiate)
                    convert_to_yolov5(this_info, annotation_path)

                # Copy the file to the new folder
                shutil.copy2(os.path.join(rootfile,file), image_path)

    print("Finished copying and cropping all relevant files over!")

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
        
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)), outline ="red")
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()

prep_yolov5_data('..\\train_val_data', 
                benign_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "benign"), 
                malig_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "malignant"),
                differentiate=differentiate_option)


# Now let's take a look at testing the annotations we made for YOLO v5
class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))


# TEST SECTION - pull up a random image and see if the annotation is correct using the YOLO format
annotations = [os.path.join('..\\train_val_data\\annotations', x) for x in os.listdir('..\\train_val_data\\annotations') if x[-3:] == "txt"]

# Get any random annotation file 
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = os.path.normpath(annotation_file.replace("annotations", "images").replace("txt", "jpg"))
print(image_file)
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)

# Separate into train, validation, and test 
images = [os.path.join('..\\train_val_data\\images', x) for x in os.listdir('..\\train_val_data\\images')]
annotations = [os.path.join('..\\train_val_data\\annotations', x) for x in os.listdir('..\\train_val_data\\annotations') if x[-3:] == "txt"]

images.sort()
annotations.sort()

# Split the original train dataset into train-validation splits
train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)

# Now pull in the test dataset
prep_yolov5_data('..\\test_data', 
                benign_folder = os.path.join(config.RAW_TEST_DATA_FOLDER, "External_test_set-benign"), 
                malig_folder = os.path.join(config.RAW_TEST_DATA_FOLDER, "External_test_set-malignant"),
                differentiate=differentiate_option)

# TEST SECTION
annotations = [os.path.join('..\\test_data\\annotations', x) for x in os.listdir('..\\test_data\\annotations') if x[-3:] == "txt"]

# Get any random annotation file 
annotation_file = random.choice(annotations)
with open(annotation_file, "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]

#Get the corresponding image file
image_file = os.path.normpath(annotation_file.replace("annotations", "images").replace("txt", "jpg"))
print(image_file)
assert os.path.exists(image_file)

#Load the image
image = Image.open(image_file)

#Plot the Bounding Box
plot_bounding_box(image, annotation_list)

# Separate into train, validation, and test 
test_images = [os.path.join('..\\test_data\\images', x) for x in os.listdir('..\\test_data\\images')]
test_annotations = [os.path.join('..\\test_data\\annotations', x) for x in os.listdir('..\\test_data\\annotations') if x[-3:] == "txt"]

test_images.sort()
test_annotations.sort()


# Create folders to hold images
folders_to_create = ["..\\yolo_data\\images\\train", 
                     "..\\yolo_data\\images\\val", 
                     "..\\yolo_data\\images\\test",
                     "..\\yolo_data\\labels\\train",
                     "..\\yolo_data\\labels\\val",
                     "..\\yolo_data\\labels\\test"]

for thisFolder in folders_to_create:
    if os.path.isdir(thisFolder):
        shutil.rmtree(thisFolder)
    Path(thisFolder).mkdir(parents=True, exist_ok=True)


#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except:
            print(f)
            assert False

# Move the splits into their correct folders
move_files_to_folder(train_images, '..\\yolo_data\\images\\train')
move_files_to_folder(val_images, '..\\yolo_data\\images\\val')
move_files_to_folder(test_images, '..\\yolo_data\\images\\test')
move_files_to_folder(train_annotations, '..\\yolo_data\\labels\\train')
move_files_to_folder(val_annotations, '..\\yolo_data\\labels\\val')
move_files_to_folder(test_annotations, '..\\yolo_data\\labels\\test')

# Move all folders to the yolov5 directory - clean out if already there
if os.path.isdir("..\\yolov5\\yolo_data"):
    shutil.rmtree("..\\yolov5\\yolo_data")
shutil.move("..\\yolo_data", "..\\yolov5")

# Clean up here!
shutil.rmtree("..\\train_val_data\\images")
shutil.rmtree("..\\train_val_data\\annotations")
shutil.rmtree("..\\test_data\\images")
shutil.rmtree("..\\test_data\\annotations")
