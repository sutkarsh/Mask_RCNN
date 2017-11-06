import random
import math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import cv2

import coco
import model as modellib

%matplotlib inline 

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your 
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


ass InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.print()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

print("Finished loading model")


# Run detection
def compute_contours(im, contour_width=5):
    results = model.detect([im], verbose=0) #run object detector
    r = results[0]
    kernel = np.ones((contour_width,contour_width),np.uint8) #kernel for morphological gradient
    contours = np.array([cv2.morphologyEx(img , cv2.MORPH_GRADIENT, kernel) for img in r['masks'].transpose(2,0,1)]).sum(axis=0)> 0 #morphological gradient
    return contours*1.0 #cast to float and return

def make_pretty_frame(im, alpha=0.6): #sorry for the function name. I was very sleepy when I wrote this
    contours = compute_contours(im)
    return alpha*im/im.max() + (1-alpha)*np.dstack([contours]*3)


