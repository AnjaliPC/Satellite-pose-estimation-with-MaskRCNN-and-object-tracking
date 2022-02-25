# Problem statement:
Satellite-pose-estimation-with-MaskRCNN-and-object-tracking
Dissertation Phase-1 mini work. (Main objective completion is in progress)  

# Overview:
Steps:
1. Dataset Preparation:  
1.1. Generate frames from the video. Separate them in the ratio 80:20 for Training and Validation respectively in folders train and val inside the folder Dataset.  
1.2. Annotate each frame in the train and val folders respectively using VGG annotation tool VGG 2.x. Download as json file and keep it in train and val folders.  
2. Training the MaskRCNN model:  
2.1. Configure parameters in the mrcnn Folder accordingly. We have extracted it from the repository:  https://github.com/matterport/Mask_RCNN  
2.2. Download a pre-trained MaskRCNN model weight file. We utilized the MSCOCO dataset weight file. Was dounloaded using the link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5  
2.3. Run the file : sample_training.py  
9. Test the model using the file: Model_testing.ipynb
  
# Team Members:
1. Anjali PC [CB.EN.P2CSE20006]
2. Uma Rani [ ]

# Team Guide:
Dr. Senthilkumar. T, Assistant Professor CSE Department, Amrita Vishwa Vidyapeetham,  Coimbatore.

# Project flow:
The Object detection model used is MaskRCNN with ResNet101 as the Backbone. The model uses pre-trained weight of MSCOCO dataset with 1000 classes.
Then we redefine the model with two classes namely Satellite and Non satellite body. The Pose estimation model initially detects the objects present init, namely Satellite or non- satellite body using MaskRCNN Detection Model and then used the detection result to feed Object tracking algorithm, deepSORT. Once the tracking is done, the out is used for forming corresponding point cloud and then used Poes estimation equations from form the respective position and attitude of the detected and tracked satellite body in the given frames.

# Performance:
We trained the model with 240 images in training set and 60 images in the validation set. We were able to obtain a mAP value of 0.993 for our validation set images for the object detection task.

# Conclusion:
Our next task is to track the detected satellite Object using deepSORT and form their point cloud. Firther we need to perform the same task of pose estimation on images with introduced occlusion situations and improve the accuracy of our model.
