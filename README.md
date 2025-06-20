This file explains the major scripts in our project and how to run the CNNs we experimented with; this file also explains how to run the YOLO model for object detection.

Explanation of Folder Structure:
Data/`  
   Contains original ultrasound training data.  
   ❗ **Excluded from GitHub** due to privacy/sensitivity.

- `DataTEST/`  
   Holds test data collected later in the semester.  
   ❗ **Excluded from GitHub** for privacy and ethical considerations.

- `pyNN/`  
   Main Python scripts for training and evaluating convolutional neural networks (CNNs).

- `pyPreprocessing_coding_snippets/`  
   Experimental code snippets and preprocessing functions used during model development.

- `Ultrasound_test/`  
   Backup of `DataTEST` directory containing test ultrasound images.  
   ❗ **Excluded from GitHub** (duplicate sensitive data).

- `yolov5/`  
   Cloned YOLOv5 repository used for object detection pipeline. Includes custom training and inference code.

The main scripts to be aware of are:
- pyNN/MAIN_script.py: This script imports our implementation of ResNet, DenseNet, and MobileNet from the CNN_wrappers file and runs them. Please see this file for a short
	       	       example of how to implement each CNN. It will also create several new folders to temporarily store the sorted training photos.
- pyNN/CNN_wrappers.py: This script implements functions that handle each type of CNN - it also defines some functions to help prepare the data
- pyNN/helper_scripts.py: Useful scripts that help train and score each model
- pyNN/YOLO_data_preparation.py: This script prepares the data for the YOLO model - this must be run before YOLOv5 is activated
- yolov5/infer_test_dataset.py: This script runs a previously trained model and scores it on the test dataset

Please note that the scripts in the pyNN folder work best if you navigate inside the folder and then run the scripts via the command line. (eg "python MAIN_script.py" inside the terminal)

The YOLOv5 model works somewhat differently - you must navigate to the yolov5 folder in your command line and input the following commands depending on what 
task you would like to accomplish:

For training:
python train.py --img 640 --cfg yolov5s.yaml --hyp ai_first_parameters.yaml --batch 32 --epochs 100 --data ai_first_data_two_classes.yaml --weights yolov5s.pt --workers 24 --name yolo_two_class_final_full_train

The arguments are:
- img: 640 (image size)
- cfg: yolov5s.yaml (which version of YOLO you would like to utilize)
- hyp: hyperparameter file - in our case we have stored our parameters in the "ai_first_parameters.yaml" -- THIS FILE IS USER CONFIGURABLE
- batch: image batch size
- epochs: number of epochs to train
- data: this file specifies the files where the data are stored (run the pyNN/YOLO_data_preparation.py script first!) -- THIS FILE IS USER CONFIGURABLE
- weights: default weights to initialize the YOLO model with
- workers: a training parameter
- name: the name you would like to give this particular run - the results will be saved in the "runs" folder

For testing:
python val.py --weights runs/train/yolo_one_class_final/weights/best.pt --data ai_first_data_one_class.yaml --task test --name yolo_test_folder_one_class

The arguments are:
- weights: The location of the .pt file that stores the best trained model from the "train" command outlined above - you can essentially store models in the .pt files and run them on a particular set of images with this command
- data: his file specifies the files where the data are stored -- THIS FILE IS USER CONFIGURABLE
- task: "test" or "val" depending on what dataset you would like to run the model on
- name: the name you would like to give this particular run - the results will be saved in the "runs" folder

As the YOLOv5 model we ran with classification turned out to be our best model, we did not build a pipeline to take the object detection results and then feed them into the base CNNs. 
It is possible, however, to extract the object detection bounding boxes from YOLOv5 and then use them to preprocess the images before feeding them to another CNN. 

For questions, you may contact us via email:
Charles Daniels - chrlsdnls301@gmail.com
Melirose Liwag - mliwag3@gatech.edu
Mehmet Gumus - mgumus3@gatech.edu
