# Specifies resNet functions

"""
Steps:
1. Prepping the Dataset
    a. Uploading the images into a data construction and labelling them as benign or malignant
2. Splitting the data into train / validation / test batches
3. Create a basic model for image classification
4. Creating the CNN model and tuning up the hyperparameters 
"""
 
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import helper_scripts
import shutil
import gc
import os
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain, combinations
import dense_implementation
from torchinfo import summary
from collections import OrderedDict


# Import local configuration
import config

# import helper scripts
from helper_scripts import sortData, display_img, crop_my_image, sortCropData, padded_sort_data

# Set the environment
torch.manual_seed(1)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def loadData(extendedPadding, resizeDimensions, normalize, batchSize, grayscale):
     # DATASET WRAPPER

    # Transform here
    sorted_data_dir = config.DATA_FOLDER
    raw_data_dir = config.RAW_DATA_FOLDER

    # Clear out the previous sorted_data folder just in case
    try:
        shutil.rmtree(sorted_data_dir)
    except:
        print("Deleting all old files...")


    # data preprocessing
    # This function creates the necessary data files - we create an extended dataset that we concatenate with the original 
    # dataset WITHOUT the extended transforms

    if extendedPadding:
        padded_sort_data(sorted_data_dir, raw_data_dir)
    else:
        sortCropData(sorted_data_dir, raw_data_dir)

    # Build transform
    if grayscale:
        transformationList = [
                #transforms.ColorJitter(contrast=0.5),
                #transforms.RandomRotation(30),
                #transforms.CenterCrop(480),
                transforms.Resize((resizeDimensions)),
                transforms.Grayscale(),
                #transforms.Pad(1),
                #transforms.Lambda(crop_my_image),
                transforms.ToTensor()
            ]
    else:
        transformationList = [
                #transforms.ColorJitter(contrast=0.5),
                #transforms.RandomRotation(30),
                #transforms.CenterCrop(480),
                transforms.Resize((resizeDimensions)),
                #transforms.Pad(1),
                #transforms.Lambda(crop_my_image),
                transforms.ToTensor()
            ]

    # Extended transform
    transform = transforms.Compose(transformationList)

    # Original no transforms
    dataset = ImageFolder(sorted_data_dir,  transform=transform) #, transforms.Compose([transforms.Resize((150,200)), transforms.ToTensor()]))


    # Use the dataloader to split up the ORIGINAL dataset
    batch_size = batchSize

    val_size = 230
    train_size = len(dataset) - val_size 
    train_data,val_data = random_split(dataset, [train_size,val_size])


    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")


    #load the train and validation into batches.
    train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size, num_workers = 0, pin_memory = True)
  
    # %% Visualize image using a make grid
    # def show_batch(dl):
    #     """Plot images grid of single batch"""
    #     for images, labels in dl:
    #         fig,ax = plt.subplots(figsize = (16,12))
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.imshow(make_grid(images,nrow=4).permute(1,2,0))
    #         break
            
    # show_batch(train_dl)

    if normalize:
        dataset = ImageFolder(sorted_data_dir,  transform=transforms.Compose(transformationList))
        data_loader = DataLoader(dataset, batch_size, shuffle = True, num_workers = 0, pin_memory = True)
        trainMean, trainStd = helper_scripts.mean_std(data_loader)

        nextTransformTrain = transformationList.copy()

        nextTransformTrain.append(transforms.Normalize(trainMean, trainStd))

        # Original no transforms
        dataset = ImageFolder(sorted_data_dir,  transform=transforms.Compose(nextTransformTrain)) #, transforms.Compose([transforms.Resize((150,200)), transforms.ToTensor()]))

        # Use the dataloader to split up the ORIGINAL dataset
        batch_size = batchSize

        train_size = len(dataset) - val_size 
        train_data,val_data = random_split(dataset,[train_size,val_size])

        train_dl = DataLoader(train_data, batch_size, shuffle = True, num_workers=0, pin_memory=True)
        val_dl = DataLoader(val_data, batch_size, num_workers=0, pin_memory=True)

    return train_dl, val_dl, trainMean, trainStd

def loadDataAugmentedDataset(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, normalize, grayscale, val_size=230):
    # extendedPadding: boolean
    # resizeDimensions: tuple = (int, int)
    # randRot: boolean
    # colorJitter: boolean
    # imgPerspective: boolean
    # batchSize: int
    # normalize: boolean
    # val_size: int
    # DATASET WRAPPER

    # Transform here
    sorted_data_dir = config.DATA_FOLDER
    raw_data_dir = config.RAW_DATA_FOLDER

    # Clear out the previous sorted_data folder just in case
    try:
        shutil.rmtree(sorted_data_dir)
        print("Deleting all old files...")
    except:
        pass



    # data preprocessing
    # This function creates the necessary data files - we create an extended dataset that we concatenate with the original 
    # dataset WITHOUT the extended transforms
    if extendedPadding:
        padded_sort_data(sorted_data_dir, raw_data_dir)
    else:
        sortCropData(sorted_data_dir, raw_data_dir)

    # Build transform
    if grayscale:
        transformationBase = [
                #transforms.ColorJitter(contrast=0.5),
                #transforms.RandomRotation(30),
                #transforms.CenterCrop(480),
                transforms.Resize((resizeDimensions)),
                transforms.Grayscale(),
                #transforms.Pad(1),
                #transforms.Lambda(crop_my_image),
                transforms.ToTensor()
            ]
    else:
        transformationBase = [
                #transforms.ColorJitter(contrast=0.5),
                #transforms.RandomRotation(30),
                #transforms.CenterCrop(480),
                transforms.Resize((resizeDimensions)),
                #transforms.Pad(1),
                #transforms.Lambda(crop_my_image),
                transforms.ToTensor()
            ]
    
  
    # %% Visualize image using a make grid
    # def show_batch(dl):
    #     """Plot images grid of single batch"""
    #     for images, labels in dl:
    #         fig,ax = plt.subplots(figsize = (16,12))
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.imshow(make_grid(images,nrow=4).permute(1,2,0))
    #         break
            
    # show_batch(train_dl)

    # If normalize is true, we need to calculate the correct statistics from above - don't include the original transforms
    # Here we ignore the other transforms!
    if normalize:
        dataset = ImageFolder(sorted_data_dir,  transform=transforms.Compose(transformationBase))
        data_loader = DataLoader(dataset, batchSize, shuffle = True, num_workers = 0, pin_memory = True)
        trainMean, trainStd = helper_scripts.mean_std(data_loader)

        nextTransformTrain = transformationBase.copy()

        nextTransformTrain.append(transforms.Normalize(trainMean, trainStd))

        # Transform with the normalization
        dataset = ImageFolder(sorted_data_dir,  transform=transforms.Compose(nextTransformTrain)) #, transforms.Compose([transforms.Resize((150,200)), transforms.ToTensor()]))

    else:
        # Original no transforms
        dataset = ImageFolder(sorted_data_dir,  transform=transforms.Compose(transformationBase)) #, transforms.Compose([transforms.Resize((150,200)), transforms.ToTensor()]))
        trainMean = 0 
        trainStd = 0

    train_size = len(dataset) - val_size 
    train_data,val_data = random_split(dataset,[train_size,val_size])

    # Assemble entire transformation list
    transformList = []
    transformOptions = []
    if randRot:
        transformOptions.append('randRot')
    if colorJitter:
        transformOptions.append('colorJitter')
    if imgPerspective:
        transformOptions.append('imgPerspective')

    allCombos = powerset(transformOptions)

    for combo in allCombos:
        if combo: # Check to see if there's a transformation in this - covers powerset EMPTY case
            thisTransform = transformationBase.copy()
 
            if ('randRot' in combo):
                thisTransform.insert(0, transforms.RandomRotation(30))
            if ('colorJitter' in combo):
                thisTransform.insert(0, transforms.ColorJitter(contrast=0.5))
            if ('imgPerspective' in combo):
                thisTransform.insert(0, transforms.RandomPerspective())

            # Extended transform
            appendThis = transforms.Compose(thisTransform)
            transformList.append(appendThis)


    for thisTransform in transformList:
        dataset_extended = ImageFolder(sorted_data_dir,transform=thisTransform)
        train_data_ext,_ = random_split(dataset,[train_size,val_size])

        train_data = torch.utils.data.ConcatDataset([train_data, train_data_ext])

    train_dl = DataLoader(train_data, batchSize, shuffle = True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(val_data, batchSize, num_workers=0, pin_memory=True)


    print(f"Length of Train Data : {len(train_data)}")
    print(f"Length of Validation Data : {len(val_data)}")
    return train_dl, val_dl, trainMean, trainStd

def loadTestData(extendedPadding, resizeDimensions, normalize, batchSize, trainMean, trainStd, grayscale):

    # Transform here
    sorted_data_dir_TEST = config.TEST_DATA_FOLDER
    raw_data_dir_TEST = config.RAW_TEST_DATA_FOLDER

    # Clear out the previous sorted_data folder just in case
    try:
        shutil.rmtree(sorted_data_dir_TEST)
        print("Deleting all old files...")
    except:
        pass


    # data preprocessing
    # This function creates the necessary data files - we create an extended dataset that we concatenate with the original 
    # dataset WITHOUT the extended transforms

    if extendedPadding:
        padded_sort_data(sorted_data_dir_TEST, raw_data_dir_TEST, benign_folder = os.path.join(config.RAW_TEST_DATA_FOLDER,  "External_test_set-benign"), malig_folder = os.path.join(config.RAW_TEST_DATA_FOLDER, "External_test_set-malignant"))
    else:
        sortCropData(sorted_data_dir_TEST, raw_data_dir_TEST, benign_folder = os.path.join(config.RAW_TEST_DATA_FOLDER, "External_test_set-benign"), malig_folder=os.path.join(config.RAW_TEST_DATA_FOLDER, "External_test_set-malignant"))

    # Build transform
    if grayscale:
        transformationList = [
                #transforms.ColorJitter(contrast=0.5),
                #transforms.RandomRotation(30),
                #transforms.CenterCrop(480),
                transforms.Resize((resizeDimensions)),
                transforms.Grayscale(),
                #transforms.Pad(1),
                #transforms.Lambda(crop_my_image),
                transforms.ToTensor()
            ]
    else:
        transformationList = [
                #transforms.ColorJitter(contrast=0.5),
                #transforms.RandomRotation(30),
                #transforms.CenterCrop(480),
                transforms.Resize((resizeDimensions)),
                #transforms.Pad(1),
                #transforms.Lambda(crop_my_image),
                transforms.ToTensor()
            ]

    dataset = ImageFolder(sorted_data_dir_TEST,  transform=transforms.Compose(transformationList)) 


    # Use the dataloader to split up the ORIGINAL dataset
    batch_size = batchSize

    print(f"Length of TEST Data : {len(dataset)}")

    #load the train and validation into batches.
    test_dl = DataLoader(dataset, batch_size, num_workers = 0, pin_memory = True)


    if normalize:
        nextTransformTrain = transformationList.copy()

        nextTransformTrain.append(transforms.Normalize(trainMean, trainStd))

        # Original no transforms
        dataset = ImageFolder(sorted_data_dir_TEST,  transform=transforms.Compose(nextTransformTrain)) #, transforms.Compose([transforms.Resize((150,200)), transforms.ToTensor()]))

        # Use the dataloader to split up the ORIGINAL dataset
        batch_size = batchSize

        test_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers=0, pin_memory=True)

    return test_dl

def resNet_wrapper(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, weightDecay, learningRate, activationFunction, normalize, epochs, grayscale=True,plot=True):

    # This function trains the ResNet function with the selected options



    def show_cache():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"{torch.cuda.memory_allocated()/(1024)} Kb")

    if randRot or colorJitter or imgPerspective:
        train_dl, val_dl, trainMean, trainStd = loadDataAugmentedDataset(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, normalize, grayscale)
    else:
        train_dl, val_dl, trainMean, trainStd = loadData(extendedPadding, resizeDimensions, normalize , batchSize, grayscale)
  
    # %% Visualize image using a make grid
    # def show_batch(dl):
    #     """Plot images grid of single batch"""
    #     for images, labels in dl:
    #         fig,ax = plt.subplots(figsize = (16,12))
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         ax.imshow(make_grid(images,nrow=4).permute(1,2,0))
    #         break
            
    # show_batch(train_dl)

    # Output train_dl, val_dl

    # %% create base NN

    # NEURAL NET PORTION


    class modelBase(nn.Module):
            
        def training_step(self, batch):
            images, labels = batch 
            images, labels = images.cuda(), labels.cuda() # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
            #(images, labels) = koila.lazy(batch)
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            images, labels = images.cuda(), labels.cuda()
            #(images, labels) = koila.lazy(batch)

            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))



    class Block(nn.Module):
        
        def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
            super(Block, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            if activationFunction == 'reLU':
                self.relu = nn.ReLU()
            else:
                self.relu = nn.ReLU6()
            self.identity_downsample = identity_downsample
            
        def forward(self, x):
            identity = x
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            if self.identity_downsample is not None:
                identity = self.identity_downsample(identity)
            x += identity
            x = self.relu(x)
            return x
        
        
        
    class ResNet_18(modelBase):
        #   GRAYSCALE
        def __init__(self, image_channels = 1, num_classes=2):

            if grayscale:
                image_channels = 1
            else:
                image_channels = 3
            
            super(ResNet_18, self).__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU6()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            #resnet layers
            self.layer1 = self.__make_layer(64, 64, stride=1)
            self.layer2 = self.__make_layer(64, 128, stride=2)
            self.layer3 = self.__make_layer(128, 256, stride=2)
            self.layer4 = self.__make_layer(256, 512, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            
        def __make_layer(self, in_channels, out_channels, stride):
            
            identity_downsample = None
            if stride != 1:
                identity_downsample = self.identity_downsample(in_channels, out_channels)
                
            return nn.Sequential(
                Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
                Block(out_channels, out_channels)
            )
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
            return x 
        
        def identity_downsample(self, in_channels, out_channels):
            
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
                nn.BatchNorm2d(out_channels)
            )
        

        
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)

    
    def fit(epochs, model, train_loader, val_loader, opt_func):
        
        history = []
        optimizer = opt_func
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


    # LOAD THE TEST DATA
    test_dl = loadTestData(extendedPadding, resizeDimensions, normalize, batchSize, trainMean, trainStd, grayscale)
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

    model = ResNet_18().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, weight_decay=weightDecay)
    # Establish a scheduler

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    resnet18_train = []
    resnet18_val = []
    restnet18_test = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainResultResNet = helper_scripts.train(train_dl, model, loss_fn, optimizer)
        valResultResNet = helper_scripts.test(val_dl, model, loss_fn)
        final_results = helper_scripts.test(test_dl, model, loss_fn, final=True)
        resnet18_train.append(trainResultResNet['trainAcc'])
        resnet18_val.append(valResultResNet['testAcc'])
        restnet18_test.append(final_results['testAcc'])

        # increment the scheduler
        scheduler.step()

    print("Done!")

    final_results = helper_scripts.test(test_dl, model, loss_fn, final=True, plotAUC=True)

    if plot:
        x = range(epochs)
        plt.figure()
        plt.plot(x, resnet18_train, label="ResNet18 TRAIN", color='red', linestyle=":")
        plt.plot(x, resnet18_val, label="ResNet18 Val", color='red',linestyle="-")
        plt.plot(x, restnet18_test, label="ResNet18 TEST", color='blue', linestyle="-")
        plt.grid()
        plt.title("ResNet18 Full Dataset Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower right')
        plt.show()
        
    return resnet18_train, resnet18_val, restnet18_test

def denseNet_wrapper(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, weightDecay, learningRate, activationFunction, normalize, epochs, grayscale=False,plot=True):
        # This function trains the ResNet function with the selected options

    def show_cache():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"{torch.cuda.memory_allocated()/(1024)} Kb")

    if randRot or colorJitter or imgPerspective:
        train_dl, val_dl, trainMean, trainStd = loadDataAugmentedDataset(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, normalize, grayscale)
    else:
        train_dl, val_dl, trainMean, trainStd = loadData(extendedPadding, resizeDimensions, normalize , batchSize, grayscale)


    # %% create base NN

    # LOAD THE TEST DATA
    test_dl = loadTestData(extendedPadding, resizeDimensions, normalize, batchSize, trainMean, trainStd, grayscale)
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

    # load the DenseNet model here
    model = dense_implementation.densenet121()
    # Use GPU if available
    if torch.cuda.is_available():
        model.cuda()


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, weight_decay=weightDecay)
    # Establish a scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    denseNet121_train = []
    denseNet121_val = []
    denseNet121_test = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainResultResNet = helper_scripts.train(train_dl, model, loss_fn, optimizer)
        valResultResNet = helper_scripts.test(val_dl, model, loss_fn)
        final_results = helper_scripts.test(test_dl, model, loss_fn, final=True)
        denseNet121_train.append(trainResultResNet['trainAcc'])
        denseNet121_val.append(valResultResNet['testAcc'])
        denseNet121_test.append(final_results['testAcc'])

        # increment the scheduler
        scheduler.step()

    print("Done!")

    final_results = helper_scripts.test(test_dl, model, loss_fn, final=True, plotAUC=True)

    if plot:
        x = range(epochs)
        plt.figure()
        plt.plot(x, denseNet121_train, label="ResNet18 TRAIN", color='red', linestyle=":")
        plt.plot(x, denseNet121_val, label="ResNet18 Val", color='red',linestyle="-")
        plt.plot(x, denseNet121_test, label="ResNet18 TEST", color='blue', linestyle="-")
        plt.grid()
        plt.title("ResNet18 Full Dataset Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower right')
        plt.show()
        
    return denseNet121_train, denseNet121_val, denseNet121_test

def mobileNetv1_wrapper(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, weightDecay, learningRate, activationFunction, normalize, epochs, grayscale=True,plot=True):
        # This function trains the ResNet function with the selected options

    def show_cache():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"{torch.cuda.memory_allocated()/(1024)} Kb")

    if randRot or colorJitter or imgPerspective:
        train_dl, val_dl, trainMean, trainStd = loadDataAugmentedDataset(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, normalize, grayscale)
    else:
        train_dl, val_dl, trainMean, trainStd = loadData(extendedPadding, resizeDimensions, normalize , batchSize, grayscale)


    # %% create base NN for mobilenet v1
    class modelBase(nn.Module):
            
        def training_step(self, batch):
            images, labels = batch 
            images, labels = images.cuda(), labels.cuda() # https://stackoverflow.com/questions/59013109/runtimeerror-input-type-torch-floattensor-and-weight-type-torch-cuda-floatte
            #(images, labels) = koila.lazy(batch)
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            images, labels = images.cuda(), labels.cuda()
            #(images, labels) = koila.lazy(batch)

            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = accuracy(out, labels)           # Calculate accuracy
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    ### MobileNet v1 architectures
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
                


    # LOAD THE TEST DATA
    test_dl = loadTestData(extendedPadding, resizeDimensions, normalize, batchSize, trainMean, trainStd, grayscale)
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu")

    # load the DenseNet model here
    model = MobileNet_v1().to(device)
    # Use GPU if available
    if torch.cuda.is_available():
        model.cuda()


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate, weight_decay=weightDecay)
    # Establish a scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    mobileNet_train = []
    mobileNet_val = []
    mobileNet_test = [] 

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        trainResultResNet = helper_scripts.train(train_dl, model, loss_fn, optimizer)
        valResultResNet = helper_scripts.test(val_dl, model, loss_fn)
        final_results = helper_scripts.test(test_dl, model, loss_fn, final=True)
        mobileNet_train.append(trainResultResNet['trainAcc'])
        mobileNet_val.append(valResultResNet['testAcc'])
        mobileNet_test.append(final_results['testAcc'])

        # increment the scheduler
        scheduler.step()

    print("Done!")

    final_results = helper_scripts.test(test_dl, model, loss_fn, final=True, plotAUC=True)

    if plot:
        x = range(epochs)
        plt.figure()
        plt.plot(x, mobileNet_train, label="ResNet18 TRAIN", color='red', linestyle=":")
        plt.plot(x, mobileNet_val, label="ResNet18 Val", color='red',linestyle="-")
        plt.plot(x, mobileNet_test, label="ResNet18 TEST", color='blue', linestyle="-")
        plt.grid()
        plt.title("ResNet18 Full Dataset Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower right')
        plt.show()
        
    return mobileNet_train, mobileNet_val, mobileNet_test