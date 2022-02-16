# Satellite-pose-estimation-with-MaskRCNN-and-object-tracking
Dissertation Phase-1 mini work. (Main objective completion is in progress)  
Steps:
1. Dataset Preparation:  
1.1. Generate frames from the video. Separate them in the ratio 80:20 for Training and Validation respectively in folders train and val inside the folder Dataset.  
1.2. Annotate each frame in the train and val folders respectively using VGG annotation tool VGG 2.x. Download as json file and keep it in train and val folders.  
2. Training the MaskRCNN model:  
2.1. Configure parameters in the mrcnn Folder accordingly. We have extracted it from the repository:  https://github.com/matterport/Mask_RCNN  
2.2. Download a pre-trained MaskRCNN model weight file. We utilized the MSCOCO dataset weight file. Was dounloaded using the link: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5  
2.3. Run the file : sample_training.py  
9. Test the model using the file: Model_testing.ipynb
  
    
Note: Many more sitation details need to be mentioned here. Will update the same quickly.
