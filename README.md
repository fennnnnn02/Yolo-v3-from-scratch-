# Yolo-v3-from-scratch

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
