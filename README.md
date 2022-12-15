# Yolo-v3-from-scratch

## Official yolov3 paper
https://arxiv.org/abs/1804.02767

## Implementing yolo v3 from scratch in pytorch

* YOLO can only detect objects belonging to the classes present in the dataset used to train the network. In this repsitory we will be using the official weight file for our detector. These weights have been obtained by training the network on COCO dataset, and therefore we can detect 80 object categories.

* yolov3 weights file downloaded from here - https://pjreddie.com/media/files/yolov3.weights

* To run detection over an image
  ```sh
  python3 detect.py
  python detect.py --images test1.png --det det

  ```
* Output will be stored in the same directory.

* Some examples of detection made are as follows

![output](https://user-images.githubusercontent.com/91083791/207788921-5d2c8ec3-4733-47ab-8901-cc0d9573232a.jpg)

![output](https://user-images.githubusercontent.com/91083791/207789414-d437fbda-0b9f-4a17-a65b-f1f85199180e.jpg)


* References followed 
  - https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
  - https://medium.datadriveninvestor.com/yolov3-from-scratch-using-pytorch-part1-474b49f7c8ef
