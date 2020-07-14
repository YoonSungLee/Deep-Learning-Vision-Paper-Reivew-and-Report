# You Only Look Once: Unified, Real-Time Object Detection

paper(15.06) https://arxiv.org/abs/1506.02640<br>
code https://pjreddie.com/darknet/yolo/

---

# Abstract

Object Detection 분야의 모델은 크게 One-Stage 계열의 YOLO 모델과, Two-Stage 계열의 R-CNN 모델로 분류할 수 있다. 그 중 YOLO 모델을 탐구함으로써, One-Stage의 원리와 그에 따른 장단점을 살펴보고자 한다.

**"we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities."**<br>
해당 시기에 Object Detection 접근법과는 달리 YOLO는 Object Detection 문제를 regression problem으로 정의하고 해결해나간다.

**"Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance."**<br>
특히 YOLO는 One-Stage의 단일 pipeline으로 구성하여 end-to-end train을 가능하도록 한 것이 특징이다.

**"YOLO makes more localization errors but is less likely to predit false positives on background."**<br>
논문의 저자는 YOLO 모델이 다른 Object Detection 모델보다 localization 성능에 있어서 더 많은 error를 일으키지만, 그 효과가 미비하다고 언급한다.

# 1. Introduction

**"We reframe object detection as a single regression problem, straight from image pixels to bounding box coordinates and class probablities."**<br>
마치 사람이 한 눈에 시야를 살펴보고 물체를 인식하는 것처럼, 기존 모델들(DPM, R-CNN)과는 달리 한 번에 classification과 localization을 수행하는 YOLO의 구조를 언급한다.

YOLO의 unifed model이라는 특징은 몇몇 장점들을 가지고 있다.<br>
<br>
**"First, YOLO is extremely fast"**<br>
YOLO는 45fps의 성능을 보이며, fast version은 150fps 이상의 성능을 보인다고 언급한다.<br>
**"Second, YOLO reasons globally about the images when making predictions."**<br>
sliding window 방식이나 region proposal-based techniques과는 달리 training동안 전체 이미지를 한 번에 본다.<br>
**"Third, YOLO learns generalizable representations of objects."**<br>

# 2. Unified Detection

**"Our system divides the input image into an SxS grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object."**<br>
YOLO는 input image를 SxS 그리드 셀로 나눈 뒤, 각 그리드 셀이 여러 개의 Bounding Box를 가지고 물체를 예측하는 구조이다. 이 때, 어떤 물체의 중심이 특정 그리드 셀에 포함된다면, 해당 그리드 셀은 그 물체를 예측할 것을 기대할 수 있다.

**"Each grid cell predicts B bounding boes and confidence scores for those boxes. These confidence scores reflect how confidence the model is that the box contains an object and also how accurate it thinks the box is that it predicts."**<br>
각 그리드 셀은 B개의 Bounding Box를 가지고 있다는 특징에 관한 설명이다. 더 자세한 구조는 아래 내용을 통해 확인할 수 있다.

**"Each bounding box consists of 5 predictions: x, y, w, h, and confidence. The (x,y) coordinates represent the center of the box relative to the bounds of the grid cell. The width and height are predicted relative to the whole image. Finally the confidence prediction represents the IOU between the predicted box and any ground truth box.<br>
Each grid cell also predicts C conditional class probabilites, Pr(Class(i)|Object)."**<br>
다시 정리하자면, YOLO는 input image를 SxS 그리드 셀로 나눈다. 그리고 각 그리드 셀은 B개의 Bounding Box와 C개의 conditional class probability를 가지고 있다. 이는 그리드 셀에 물체가 있다고 가정했을 때, 각 class 별로 존재할 확률을 의미한다. 그리고 각 Bounding Box는 x, y, w, h, confidence를 가지고 있다. 따라서 예측에 사용되는 feature map은 S x S x ( Bx5 + C) tensor로 구성되어 있다는 것을 확인할 수 있다. 주의할 점은 각 Bounding Box별로 conditional class probability를 가지고 있는 것이 아니라, 각 그리드 셀 별로 가지고 있다는 점이다. 즉, 특정 그리드 셀 내의 모든 Bounding Box는 conditional class probability에 대한 정보를 공유하고 있다.

### 2.1. Network Design

<img src = 'https://i.imgur.com/BL31OQu.png' width='100%'>

* Convolutional neural network
* PASCAL VOC detection dataset
* 24 convolutional layers + 2 fully connected layers
* 1x1 reduction layers

### 2.2. Training

Architecture
* The first 20 convolutional layers --> pretrain on the ImageNet 1000-class competition dataset
* Add 4 convolutional layers and 2 fully connected layers with randomly initialized weights
* From 224x224 to 448x448 <-- because detection often requires fine-grained visual information

Normalization x, y, w, h
* Normalize the bounding box width and height by the image width and height so that they fall between 0 and 1
* Parameterize the bounding box x and y coordinates to be offsets of a particular grid cell location so they are also bounded between 0 and 1

Activation function
* Linear activation function for the final layer
* Leaky rectified linear activation function for all other layers

Optimization
* Sum-squared error <-- it is easy to optimize
* Increase the loss from bounding box coordinate preditions and decrease the loss from confidence predictions for boxes that don't contain objects
* To partially address this we predict the square root of the bounding box width and height instead of the width and height directly.

IOU
* We assign one predictor to be 'responsible' for predicting an object based on which prediction has the highest current IOU with the ground truth.

Loss function

<img src='https://i.imgur.com/zZiXI4s.png' width='100%'>

Training
* batch size 64
* momentum 0.9
* decay 0.0005
* learning rate scheduler
* dropout 0.5 after the first connected layer <-- prevent co-adaptation between layers
* data agumentation: random scaling, translation of up to 20% of the original image size

### 2.3. Inference

**"The grid design enforces spatial diversity in the bounding box prediction."**<br>
YOLO의 그리드 셀 접근 방식은 Bounding Box Regression에 있어서 공간적인 다양성을 증가시킨다. 일반적으로 각 object는 하나의 그리드 셀 안에 할당되며, 해당 그리드 셀 내의 하나의 bounding box에 할당된다. 하지만 큰 물체나 여러 그리드 셀의 가장자리에 걸쳐져 있는 object의 경우, 여러 개의 그리드 셀에 할당되고, 마찬가지로 여러 개의 bounding box에 할당될 것이다. 이러한 경우에 NMS(Non-Maximal Suppression)이 큰 효과를 줄 수 있다고 언급하고 있다.

### 2.4. Limitations of YOLO

**"YOLO imposes strong spatial constraints on bounding bo predictions since each grid cell only predicts two boxes and can only have one class.""**<br>
YOLO의 그리드 셀 접근 방식은 공간적인 제약을 만들어낸다. 각 그리드 셀의 각 Bounding Box는 해당 위치를 바탕으로 오직 하나의 class만 예측하기 때문이다. 이러한 제약은 물체들이 모여 있을 때, 그 단점이 드러난다. 예를 들어, 각 그리드 셀 안에서 예측하는 Bounding Box가 새 무리를 모두 감지할 수는 없을 것이다.

**"Since our model learns to predict bounding boxes from data, it struggles to generalize to object in new or unusual aspect ratios or configurations."**<br>
YOLO 모델은 새로운 비율이나 새로운 형태에 대한 object를 맞추는 것에 약한 특성을 가지고 있다.

*discussion*<br>
**"Finally, while we train on a loss function that approximates detection performance, our loss function treats errors the same in small bounding boxes versus large bounding boxes."**<br>
loss function을 통해 학습을 할 때, 큰 Bounding Box에 대한 loss 값과 작은 Bounding Box에 대한 loss 값이 동일하게 부여되기 때문에 문제가 생긴다고 언급한다. 내 의견으로는 Bounding Box Regression하는 과정에 있어서 x, y, w, h가 0~1로 normalize되기 때문에 큰 문제가 없다고 생각했는데, 좀 더 확인해 볼 필요가 있다고 판단한다.

# 3. Comparison to Other Detection System

(...skip...)

# 4. Experiments

(...skip...)

# 5. Real-Time Detection In The Wild

**"We connect YOLO to a webcam and verify that it maintains real-time performance, including the time to fetch images from the camera and display the detections."**<br>
YOLO는 One-Stage Detection Model로써, Real-time performance가 가능하다. 따라서 실제로 webcam을 통해 그 성능을 낼 수 있다는 설명이다.

# 6. Conclusion

(...skip...)
