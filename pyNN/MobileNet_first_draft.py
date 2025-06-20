# first draft at building a MobileNetV1 to classify

"""
Steps:
1. Prepping the Dataset
    a. Uploading the images into a data construction and labelling them as benign or malignant
2. Splitting the data into train / validation / test batches
3. Create a basic model for image classification
4. Creating the CNN model and tuning up the hyperparameters 

Stuff to watch for:
- CHECK YOUR RESHAPE - not all images are the same size
- INSIDE THE LINEAR FUNCTION make sure the dimensions for the layers make sense

Useful sources:
- Preparing images: https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
- ImageFolder class for creating a useful dataset: https://debuggercafe.com/pytorch-imagefolder-for-training-cnn-models/


skeleton: https://medium.com/thecyphy/train-cnn-model-with-pytorch-21dafb918f48

"""
#%%
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import shutil
import gc
import os

# Import local configuration
import config

# import helper scripts
from helper_scripts import sortData, display_img, crop_my_image, sortCropData

torch.manual_seed(1)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
#%%
# Clear out the cache for a hardware accelerated run
def show_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"{torch.cuda.memory_allocated()/(1024)} Kb")

# Transform here
sorted_data_dir = config.DATA_FOLDER
raw_data_dir = config.RAW_DATA_FOLDER

# Clear out the previous sorted_data folder just in case
shutil.rmtree(sorted_data_dir)

# data preprocessing
# This function creates the necessary data files


sortCropData(sorted_data_dir, raw_data_dir)


# Example transform
transform = transforms.Compose(
    [
        #transforms.ColorJitter(contrast=0.5),
        #transforms.RandomRotation(30),
        #transforms.CenterCrop(480),
        transforms.Resize((150,200)),
        transforms.Pad(1),
        #transforms.Lambda(crop_my_image),
        transforms.ToTensor()
    ]
)
# How to AUGMENT a transformed dataset to the original dataset
# https://stackoverflow.com/questions/70953156/applying-transformation-to-data-set-in-pytorch-and-add-them-to-the-data

# Custom transform here
# https://medium.com/@sergei740/simple-guide-to-custom-pytorch-transformations-d6bdef5f8ba2

# Cropping function
#https://pytorch.org/vision/stable/generated/torchvision.transforms.functional.crop.html#torchvision.transforms.functional.crop

# Cast Images into a dataclass here
# Load images here
dataset = ImageFolder(sorted_data_dir, transform=transform) #, transforms.Compose([transforms.Resize((150,200)), transforms.ToTensor()]))

#%%
# test directory to come


# Did the dataset load correctly?
img, label = dataset[0]
print(img.shape, label)
print(dataset.classes)



#display the first image in the dataset
display_img(*dataset[0], dataset)



# %% Use the dataloader to split up the dataset
batch_size = 16
val_size = 200
train_size = len(dataset) - val_size 

train_data,val_data = random_split(dataset,[train_size,val_size])
print(f"Length of Train Data : {len(train_data)}")
print(f"Length of Validation Data : {len(val_data)}")


#load the train and validation into batches.
train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
val_dl = DataLoader(val_data, batch_size*2, num_workers = 0, pin_memory = True)

# %% Visualize image using a make grid
def show_batch(dl):
    """Plot images grid of single batch"""
    for images, labels in dl:
        fig,ax = plt.subplots(figsize = (16,12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images,nrow=4).permute(1,2,0))
        break
        
show_batch(train_dl)

# %% create base NN

import torch.nn as nn
import torch.nn.functional as F

class modelBase(nn.Module):
        
    def training_step(self, batch):
        images, labels = batch 
        #images, labels = images.cuda(), labels.cuda() # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
        #(images, labels) = koila.lazy(batch)
        out = self(images)                  # Generate predictions
        print("Target Shape: ", labels.shape)
        print("Input Shape: ", out.shape)
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        #images, labels = images.cuda(), labels.cuda()
        #(images, labels) = koila.lazy(batch)

        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()           # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()              # Combine accuracies

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

### MobileNet v1 architectures
from torchinfo import summary
from collections import OrderedDict

class Depthwise_conv(nn.Module):
    '''
    Architecture:
    #  3x3 Depthwise Conv
    #  BN
    #  ReLU
    '''
    def __init__(self, in_fts, stride=(1,1)):
        super(Depthwise_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, in_fts, kernel_size=(3,3), stride=stride,
                      padding=(1,1), groups=in_fts),
            nn.BatchNorm2d(in_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

class Pointwise_conv(nn.Module):
    '''
    Architecture:
    #  1x1 Conv
    #  BN
    #  ReLU
    '''
    def __init__(self, in_fts, out_fts):
        super(Pointwise_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, out_fts, kernel_size=(1,1)),
            nn.BatchNorm2d(out_fts),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_image):
        x = self.conv(input_image)
        return x

class Depthwise_Separable_conv(nn.Module):
    '''
    Architecture:
    #  3x3 Depthwise Conv
    #  BN
    #  ReLU
    #  1x1 Conv
    #  BN
    #  ReLU
    '''
    def __init__(self, in_fts, out_fts, stride=(1,1)):
        super(Depthwise_Separable_conv, self).__init__()
        self.dw = Depthwise_conv(in_fts=in_fts, stride=stride)
        self.pw = Pointwise_conv(in_fts=in_fts, out_fts=out_fts)

    def forward(self, input_image):
        x = self.pw(self.dw(input_image))
        return x

class MobileNet_v1(modelBase):
    def __init__(self, in_fts=3, num_filter=32):
        super(MobileNet_v1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_fts, num_filter, kernel_size=(3,3), stride=(2,2),
                     padding=(1,1)),
            nn.BatchNorm2d(num_filter),
            nn.ReLU(inplace=True)
        )

        self.in_fts = num_filter
        self.nlayer_filter = [
            num_filter * 2,
            [num_filter * pow(2,2)],
            num_filter * pow(2,2),
            [num_filter * pow(2,3)],
            num_filter * pow(2,3),
            [num_filter * pow(2,4)],
            [5, num_filter * pow(2,4)],
            [num_filter * pow(2,5)],
            num_filter * pow(2,5)
        ]

        self.DSC = self.layer_construct()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Sequential(
            nn.Linear(1024,1000),
            nn.Softmax(dim=1)
        )

    def forward(self, input_image):
        N = input_image.shape[0]
        x = self.conv(input_image)
        x = self.DSC(x)
        x = self.avgpool(x)
        x = x.reshape(N, -1)
        x = self.fc(x)
        return x

    def layer_construct(self):
        block = OrderedDict()
        index = 1
        for l in self.nlayer_filter:
            if type(l) == list:
                if len(l) == 2:
                    for _ in range(l[0]):
                        block[str(index)] = Depthwise_Separable_conv(self.in_fts, l[1])
                        index += 1
                else:
                    block[str(index)] = Depthwise_Separable_conv(self.in_fts, l[0], stride=(2,2))
                    self.in_fts = l[0]
                    index += 1
            else:
                block[str(index)] = Depthwise_Separable_conv(self.in_fts, l)
                self.in_fts = l
                index += 1
        return nn.Sequential(block)
            
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

  
def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history

#%%
import time 
def main():
    num_epochs = 30
    opt_func = torch.optim.Adam
    lr = 0.001 #fitting the model on training data and record the result after each epoch
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    startTime = time.time()

    model = MobileNet_v1().to(device)
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    summary(model, (1,3,224,224))

    endTime = time.time()
    print("Runtime: ", endTime - startTime)
# %%
if __name__== '__main__':
    main()
# %%
