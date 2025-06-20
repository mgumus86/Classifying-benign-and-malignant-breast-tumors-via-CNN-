# This script extracts and sorts the data into a benign / malignant folder (and adds a prefix indicating the classification of the image)
#%%
import os 
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import config
import json
import math
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import glob
import cv2
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd



# Show an image!
def display_img(img,label,dataset):
    
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))

# function to calculate average / standard deviation of train data
def mean_std(data_loader):
    this_sum = 0
    sq_sum = 0
    num_batches = 0
    for data, _ in data_loader:
        this_sum += torch.mean(data, dim=[0,2,3])
        sq_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    data_mean = this_sum/num_batches
    data_std = (sq_sum/num_batches - data_mean**2)**0.5
    return data_mean, data_std 

def crop_my_image(image: torch.tensor) -> torch.tensor:
    """Crop the images so only a specific region of interest is shown to my PyTorch model"""
    left, top, width, height = 300, 90, 100, 100
    return transforms.functional.crop(image, left=left, top=top, width=width, height=height)

def padded_sort_data(DATA_DIR, RAW_DATA_DIR, benign_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "benign"), malig_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "malignant")):
    # Sort the images and pad the images irrepsective of original image size

    dir_list = [f for f in os.listdir(malig_folder) if not f.startswith('.')]

    Tumor_list_malignant = []
    cropped_images_malignant = []

    diff1 = []
    diff2 = []

    for id in dir_list:
        output_path1 = os.path.join(malig_folder, id, '*.jpg') #r'Ultrasound-labeled/malignant/'+id+'/*.jpg'
        
        images = []
        labels = []
    
        for file in glob.glob(output_path1):

            images.append(cv2.imread(file))
            output_path2 = file[:-4]+'.json'
            with open(output_path2) as json_file:
                label = json.load(json_file)
            labels.append(label)
            
            position = label['shapes'][0]['points']
            pos1 = math.floor(position[0][1])
            pos2 = math.ceil(position[1][1])
            pos3 = math.floor(position[0][0])
            pos4 = math.ceil(position[1][0])
        
            d1 = pos2-pos1
            d2 = pos4-pos3
            
            n1 = int(365 - d1/2)
            n2 = int(365 + d1/2)
            
            n3 = int(492 - d2/2)
            n4 = int(492 + d2/2)
            
            padded_image = np.zeros((730, 983, 3), np.uint8)
            padded_image[n1:n2, n3:n4, :] = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[pos1:pos2, pos3:pos4,:]
            #padded_image[n1:n2, n3:n4, :] = cv2.imread(file)[pos1:pos2, pos3:pos4,:]
            
            cropped_images_malignant.append(padded_image)
            
            diff1.append(pos2-pos1) # maximum difference is 730
            diff2.append(pos4-pos3) # maximum difference is 983
            
        Tumor = {'Type': 'Malignant', 'Patient': {'id': id, 'Image': images, 'Tumor_Label':  labels[0]['shapes'][0]['points']}}
        Tumor_list_malignant.append(Tumor)

    Tumor_list_benign = []
    cropped_images_benign = []

    #Benign Images
    dir_list = [f for f in os.listdir(benign_folder) if not f.startswith('.')]

    Tumor_list_benign = []
    cropped_images_benign = []

    for id in dir_list:
        output_path1 = os.path.join(benign_folder, id, '*.jpg') #r'Ultrasound-labeled/benign/'+id+'/*.jpg'
        
        images = []
        labels = []
        
        for file in glob.glob(output_path1):
            images.append(cv2.imread(file))
            output_path2 = file[:-4]+'.json'
            with open(output_path2) as json_file:
                label = json.load(json_file)
            labels.append(label)
            
            position = label['shapes'][0]['points']
            pos1 = math.floor(position[0][1])
            pos2 = math.ceil(position[1][1])
            pos3 = math.floor(position[0][0])
            pos4 = math.ceil(position[1][0])
            
            d1 = pos2-pos1
            d2 = pos4-pos3
            
            n1 = int(365 - d1/2)
            n2 = int(365 + d1/2)
            
            n3 = int(492 - d2/2)
            n4 = int(492 + d2/2)
            
            padded_image = np.zeros((730, 983, 3), np.uint8)
            padded_image[n1:n2, n3:n4, :] = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)[pos1:pos2, pos3:pos4,:]

            #padded_image[n1:n2, n3:n4, :] = cv2.imread(file)[pos1:pos2, pos3:pos4,:]
            
            cropped_images_benign.append(padded_image)
            
            diff1.append(pos2-pos1) # maximum difference is 730
            diff2.append(pos4-pos3) # maximum difference is 983
        
        Tumor = {'Type': 'Benign', 'Patient': {'id': id, 'Image': images, 'Tumor_Label':  labels[0]['shapes'][0]['points']}}
        Tumor_list_benign.append(Tumor)

    # Save files in sorted_data folder
    # If a sorted_data folder doesn't exist, make it
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Create a benign folder
    benignFolder = os.path.join(DATA_DIR, 'benign')
    Path(benignFolder).mkdir(parents=True, exist_ok=True)

    # Create a malignant folder
    maligFolder = os.path.join(DATA_DIR, 'malig')
    Path(maligFolder).mkdir(parents=True, exist_ok=True)

    for k in range(len(cropped_images_malignant)):
        cv2.imwrite(os.path.join(maligFolder,'im'+str(k)+'.jpeg'), cropped_images_malignant[k])

    for k in range(len(cropped_images_benign)):
        cv2.imwrite(os.path.join(benignFolder,'im'+str(k)+'.jpeg'), cropped_images_benign[k])
    
    return cropped_images_benign


def sortCropData(DATA_DIR, RAW_DATA_DIR, benign_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "benign"), malig_folder = os.path.join(config.RAW_DATA_FOLDER, "Ultrasound-labeled", "malignant")):
    # This function takes the original photo folder and sorts them into a new folder with benign / malicious classifications
    # AND it also crops them according to what's inside the JSON files
    # Create a set that stores all file types
    types = set()

    # If a sorted_data folder doesn't exist, make it
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Create a benign folder
    benignFolder = os.path.join(DATA_DIR, 'benign')
    Path(benignFolder).mkdir(parents=True, exist_ok=True) 

    # Create a malignant folder
    maligFolder = os.path.join(DATA_DIR, 'malig')
    Path(maligFolder).mkdir(parents=True, exist_ok=True)

    # Loop through the Data folder and pull the benign photos into the benign folder
    benignCount = 0
    for rootfile, dirs, files in os.walk(benign_folder):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                # Read in accompanying JSON file here
                jsonPath = file[:-4]+'.json'
                with open(os.path.join(rootfile,jsonPath)) as json_file:
                    label = json.load(json_file)
                
                position = label['shapes'][0]['points']
                y_min = math.floor(position[0][1])
                y_max = math.ceil(position[1][1])
                x_min = math.floor(position[0][0])
                x_max = math.ceil(position[1][0])

                # Read in file to image here
                thisImg = Image.open(os.path.join(rootfile, file))

                # Crop image
                croppedImg = thisImg.crop((x_min, y_min, x_max, y_max))

                # Save to the benignFolder
                croppedImg.save(os.path.join(benignFolder, file))
                
                benignCount += 1



    # Do the same for the malignant folder
    maligCount = 0
    for rootfile, dirs, files in os.walk(malig_folder):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                # Read in accompanying JSON file here
                jsonPath = file[:-4]+'.json'
                with open(os.path.join(rootfile,jsonPath)) as json_file:
                    label = json.load(json_file)
                
                position = label['shapes'][0]['points']
                y_min = math.floor(position[0][1])
                y_max = math.ceil(position[1][1])
                x_min = math.floor(position[0][0])
                x_max = math.ceil(position[1][0])

                # Read in file to image here
                thisImg = Image.open(os.path.join(rootfile, file))

                # Crop image
                croppedImg = thisImg.crop((x_min, y_min, x_max, y_max))

                # Save to the maligFolder
                croppedImg.save(os.path.join(maligFolder, file))
                maligCount += 1


    # Rename everything in the benign folder to start with benign
    i = 0
    for filename in os.listdir(benignFolder):
        os.rename(os.path.join(benignFolder, filename), os.path.join(benignFolder, 'benign_'+filename[:-4]+'.jpg'))
        i+=1

    # Repeat with malig
    i = 0
    for filename in os.listdir(maligFolder):
        os.rename(os.path.join(maligFolder, filename), os.path.join(maligFolder, 'malig_'+filename[:-4]+'.jpg'))
        i+=1

    print("Finished copying and cropping all relevant files over!")
    print("All files types:")
    print(types)
    print(f"Benign photo count: {benignCount}")
    print(f"Malignant photo count: {maligCount}")

def sortData(DATA_DIR, RAW_DATA_DIR):
    # This function takes the original photo folder and sorts them into a new folder with benign / malicious classifications
    # Create a set that stores all file types
    types = set()

    # If a sorted_data folder doesn't exist, make it
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Create a benign folder
    benignFolder = os.path.join(DATA_DIR, 'benign')
    Path(benignFolder).mkdir(parents=True, exist_ok=True)

    # Create a malignant folder
    maligFolder = os.path.join(DATA_DIR, 'malig')
    Path(maligFolder).mkdir(parents=True, exist_ok=True)

    # Loop through the Data folder and pull the benign photos into the benign folder
    benignCount = 0
    for rootfile, dirs, files in os.walk(os.path.join(RAW_DATA_DIR, "Ultrasound-labeled", "benign")):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                shutil.copy(os.path.join(rootfile,file), benignFolder)
                benignCount += 1



    # Do the same for the malignant folder
    maligCount = 0
    for rootfile, dirs, files in os.walk(os.path.join(RAW_DATA_DIR, "Ultrasound-labeled", "malignant")):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                shutil.copy(os.path.join(rootfile,file), maligFolder)
                maligCount += 1


    # Rename everything in the benign folder to start with benign
    i = 0
    for filename in os.listdir(benignFolder):
        os.rename(os.path.join(benignFolder, filename), os.path.join(benignFolder, 'benign_'+filename[:-4]+'.jpg'))
        i+=1

    # Repeat with malig
    i = 0
    for filename in os.listdir(maligFolder):
        os.rename(os.path.join(maligFolder, filename), os.path.join(maligFolder, 'malig_'+filename[:-4]+'.jpg'))
        i+=1

    print("Finished copying all relevant files over!")
    print("All files types:")
    print(types)
    print(f"Benign photo count: {benignCount}")
    print(f"Malignant photo count: {maligCount}")

def train(dataloader, model, loss_fn, optimizer):
    # Train the model on the current data

    size = len(dataloader.dataset)
    model.train()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Calculate training accuracy
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            truePositive += ((pred.argmax(1) == y) & (y==1)).type(torch.float).sum().item()
            trueNegative += ((pred.argmax(1) == y) & (y==0)).type(torch.float).sum().item()
            falsePositive += ((pred.argmax(1) == 1) & (y==0)).type(torch.float).sum().item()
            falseNegative += ((pred.argmax(1) == 0) & (y==1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if (truePositive + falsePositive > 0):
        precision = truePositive / (truePositive + falsePositive)
    else:
        precision = 0.0

    if (truePositive + falseNegative > 0):
        recall = truePositive / (truePositive + falseNegative)
    else:
        recall = 0.0

    if (precision+recall > 0):
        f1 = 2*((precision*recall)/(precision+recall))
    else:
        f1 = 0.0

    results = {'trainAcc': correct,
               'trainPrec': precision,
               'trainRecall': recall,
               'trainf1': f1}
    return results

def test(dataloader, model, loss_fn, final=False, plotAUC=False):
    # This function outputs the f1, precision, recall, and accuracy scores - as well as some graphs

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    rawProbability = []
    correctLabels = []
    finalPrediction = []
    test_loss, correct = 0, 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            truePositive += ((pred.argmax(1) == y) & (y==1)).type(torch.float).sum().item()
            trueNegative += ((pred.argmax(1) == y) & (y==0)).type(torch.float).sum().item()
            falsePositive += ((pred.argmax(1) == 1) & (y==0)).type(torch.float).sum().item()
            falseNegative += ((pred.argmax(1) == 0) & (y==1)).type(torch.float).sum().item()
            rawProbability.extend((F.softmax(pred, dim=1)[:,1]).cpu().numpy())
            correctLabels.extend(y.cpu().numpy())
            finalPrediction.extend(pred.argmax(1).cpu())



    test_loss /= num_batches
    correct /= size

    if final:
        print(f"FINAL Error on Test Dataset: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    else:
        print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
    if (truePositive + falsePositive > 0):
        precision = truePositive / (truePositive + falsePositive)
    else:
        precision = 0.0

    if (truePositive + falseNegative > 0):
        recall = truePositive / (truePositive + falseNegative)
    else:
        recall = 0.0

    if (precision+recall > 0):
        f1 = 2*((precision*recall)/(precision+recall))
    else:
        f1 = 0.0
    results = {'testAcc': correct,
               'testPrec': precision,
               'testRecall': recall,
               'testf1': f1,
               'rawProbability': rawProbability}
    
    if plotAUC:
        print(f"AUC Score: {roc_auc_score(correctLabels, rawProbability)}")
        nn_fpr, nn_tpr, nn_thresholds = roc_curve(correctLabels, rawProbability)
        plt.figure()
        plt.plot(nn_fpr,nn_tpr,marker='.')
        plt.title('Receiver Operating Characteristic')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.grid()
        plt.show()

        cf_matrix = confusion_matrix(correctLabels, finalPrediction)

        classes =  ('Benign', 'Malignant')
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])

        plt.figure()
        sn.heatmap(df_cm, annot=True)
        plt.title("Confusion Matrix")


        # Heatmap for threshold 0.25

        # Find optimal threshold
        maxAcc = 0
        bestThresh = 0
        corrLabels = np.array(correctLabels)
        rawProb = np.array(rawProbability)
        for i in range(1,100):
            if maxAcc < (sum(corrLabels == (rawProb>(i/100)))/len(corrLabels)):
                bestThresh = i
                maxAcc = (sum(corrLabels == (rawProb>(i/100)))/len(corrLabels))
        print(f"Best Threshold is {bestThresh}%")
        print(f"Best Accuracy is {maxAcc}")


        plt.figure()
        cf_matrix = confusion_matrix(correctLabels, [ind>(bestThresh/100) for ind in rawProbability])
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes], columns = [i for i in classes])
        sn.heatmap(df_cm, annot=True)
        plt.title(f"Confusion Matrix (Threshold {bestThresh/100})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
    
    return results

# %%
