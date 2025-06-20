# This script extracts and sorts the data into a benign / malignant folder (and adds a prefix indicating the classification of the image)

import os 
from pathlib import Path
import shutil

def sortData():
    # Create a set that stores all file types
    types = set()

    # If a sorted_data folder doesn't exist, make it
    Path(r"..\sorted_data").mkdir(parents=True, exist_ok=True)

    # Create a benign folder
    benignFolder = r"..\sorted_data\benign"
    Path(benignFolder).mkdir(parents=True, exist_ok=True)

    # Create a malignant folder
    maligFolder = r"..\sorted_data\malig"
    Path(maligFolder).mkdir(parents=True, exist_ok=True)

    # Loop through the Data folder and pull the benign photos into the benign folder
    benignCount = 0
    for rootfile, dirs, files in os.walk(r".\Data\Ultrasound - AI+ First\Ultrasound\benign"):
        # Loop through the files now
        for file in files:
            splitFile = os.path.splitext(file)
            types.add(splitFile[-1])

            if file.endswith('.jpg'):
                shutil.copy(os.path.join(rootfile,file), benignFolder)
                benignCount += 1



    # Do the same for the malignant folder
    maligCount = 0
    for rootfile, dirs, files in os.walk(r".\Data\Ultrasound - AI+ First\Ultrasound\malignant"):
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