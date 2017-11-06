## Requirements
* Python 3.4+
* TensorFlow 1.3+
* Keras 2.0.8+
* Jupyter Notebook
* Numpy, skimage, scipy, Pillow
* Opencv for computing contours from mask (optional. Original code just produces mask)


## Instructions
* Download the pretrained weights file from [here](https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5) into this directory.
* install pycocotools (a util library for COCO dataset for Python. Unfortunately required to make the model work in its current state, but I will remove it in future) using the setup.py file in cocoapi/pycocotools. You might need to copy the folder pycocotools into this directory after compilation.
* Run demo.ipynb
* main.py has the same code as demo.ipynb, but in .py

