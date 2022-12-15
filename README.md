# Yolo-v3-from-scratch

## Implementing yolo v3 from scratch in pytorch

* YOLO can only detect objects belonging to the classes present in the dataset used to train the network. In this repsitory we will be using the official weight file for our detector. These weights have been obtained by training the network on COCO dataset, and therefore we can detect 80 object categories.

* yolov3 weights file downloaded from here - https://pjreddie.com/media/files/yolov3.weights

* To run detection over an image
  ```sh
  python3 detect.py
  ```
* Output will be stored in the same directory.

* Some examples of detection made are as follows
