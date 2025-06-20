# resNet Test Bench

from CNN_wrappers import resNet_wrapper, denseNet_wrapper, mobileNetv1_wrapper
import numpy as np
import shutil


# Hyperparameters
extendedPadding = True
resizeDimensions = (250, 250)
randRot = False
colorJitter = False
imgPerspective = False
batchSize = 16
weightDecay = 0.001 
learningRate = 0.01 
activationFunction = 'reLU' # Other option: 'reLU6'
epochs = 3 #60
normalize = True 
grayscale =  False 

# NOTE: The plot option will turn on the plots showing the training history
# NOTE: Training and validation datasets come from the original dataset; the test dataset metrics come from the final test dataset

# RESNET
resnet18_train, resnet18_val, resnet18_test = resNet_wrapper(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, weightDecay, learningRate, activationFunction, normalize, epochs, grayscale=grayscale, plot=True)

# NOTE: Grayscale = True does not work for DenseNet 
# DENSENET
denseNet121_train, denseNet121_val, denseNet121_test = denseNet_wrapper(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, weightDecay, learningRate, activationFunction, normalize, epochs, grayscale=grayscale, plot=True)

# NOTE: Grayscale = True does not work for MobileNet
# MOBILENET
mobileNet_train, mobileNet_val, mobileNet_test = mobileNetv1_wrapper(extendedPadding, resizeDimensions, randRot, colorJitter, imgPerspective, batchSize, weightDecay, learningRate, activationFunction, normalize, epochs, grayscale=grayscale, plot=True)

# Clean up the temporary files
shutil.rmtree("..\\sorted_data")
shutil.rmtree("..\\sorted_data_test")

