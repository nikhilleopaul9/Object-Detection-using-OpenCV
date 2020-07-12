# Object-Detection-using-OpenCV
Image and Real Time Object Detection using the most popular object detection frameworks YOLO, SSD and Mask-RCNN. OpenCV dnn module supports running inference on pre-trained deep learning models from popular frameworks like Caffe, Torch and TensorFlow.

When it comes to object detection, popular detection frameworks are:

> **YOLO**

> **SSD**

> **Mask R-CNN**

## YOLO(You only Look Once)

YOLO divides each image into a grid of S x S and each grid predicts N bounding boxes and confidence. The confidence reflects the accuracy of the bounding box and whether the bounding box actually contains an object(regardless of class). YOLO also predicts the classification score for each box for every class in training. You can combine both the classes to calculate the probability of each class being present in a predicted box. So, total SxSxN boxes are predicted. However, most of these boxes have low confidence scores and if we set a threshold say 30% confidence, we can remove most of them as shown in the example below.

The model has several advantages over classifier-based systems. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast, more than 1000x faster than R-CNN and 100x faster than Fast R-CNN.

## Single Shot Detector(SSD)

Single Shot Detector achieves a good balance between speed and accuracy. SSD runs a convolutional network on input image only once and calculates a feature map. Now, we run a small 3×3 sized convolutional kernel on this feature map to predict the bounding boxes and classification probability. SSD also uses anchor boxes at various aspect ratio similar to Faster-RCNN and learns the off-set rather than learning the box. In order to handle the scale, SSD predicts bounding boxes after multiple convolutional layers. Since each convolutional layer operates at a different scale, it is able to detect objects of various scales.

That’s a lot of algorithms. Which one should you use? Currently, Faster-RCNN is the choice if you are fanatic about the accuracy numbers. However, if you are strapped for computation(probably running it on Nvidia Jetsons), SSD is a better recommendation. Finally, if accuracy is not too much of a concern but you want to go super fast, YOLO will be the way to go.

## Mask R-CNN

Our approach efficiently detects objects in an image while simultaneously generating a high-quality segmentation mask for each instance. The method, called Mask R-CNN, extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition. Mask R-CNN is simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Moreover, Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework. We show top results in all three tracks of the COCO suite of challenges, including instance segmentation, bounding-box object detection, and person keypoint detection. Without bells and whistles, Mask R-CNN outperforms all existing, single-model entries on every task, including the COCO 2016 challenge winners.

## Model Training

* SSD: Performs Caffe-based MobileNet SSD object detection on 20 COCO classes with CUDA.

* YOLO: Performs YOLO V3 object detection on 80 COCO classes with CUDA.

* Mask-RCNN: Performs TensorFlow-based Inception V2 segmentation on 90 COCO classes with CUDA.

Each of the model files and class name files are included in their respective folders with the exception of our MobileNet SSD (the class names are hardcoded in a Python list directly in the script).

*Note: we will use OpenCV’s DNN module compiled with CUDA support. If your OpenCV is not compiled with CUDA support for your NVIDIA GPU, then you need to configure your system*

## Result for YOLO

![Screenshot (41)](https://user-images.githubusercontent.com/49313619/87250954-f2b71280-c485-11ea-8ba0-3d2b3a9bc525.png)

![Screenshot (44)](https://user-images.githubusercontent.com/49313619/87250957-f5196c80-c485-11ea-9bc3-0cee3fd82263.png)
