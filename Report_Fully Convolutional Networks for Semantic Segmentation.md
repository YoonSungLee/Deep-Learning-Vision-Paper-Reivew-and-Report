# Fully Convolutional Networks for Semantic Segmentation

---

# 1. Segmentation
* supervised learning
* downsampling&upsampling
* pixel 별 classification
* In order to predict what is in the input for each pixel, segmentation needs to recover not only what is in the input, but also where
* 만일 특정 픽셀이 어떤 클래스에도 해당하지 않는 경우, background 클래스로 규정해 0을 표기하는 방식 사용

# 2. Characteristic
* FCN uses a convolutional neural network to transform image pixels to pixel categories
* Network designed with all convolutional layers, with down-sampling and up-sampling operations
    * Downsampling path: capture semantic/contextual information, to extract and interpret the context(what)
    * Upsampling path: recover spatial information, to enable precise localization(where)
* Given a position on the spatial dimension, the output of the channel dimension will be a category prediction of the pixel corresponding to the location
* Defined the space of fully convolutional networks<br>
이미지의 spatial dimension을 유지하기 위해 기존 classification model에 fully connected layer 대신 fully convolutional layer를 적용한다.
* Defined a skip architecture
* FCN + max(AlexNet/VGG16/GoogLeNet) = FCN + VGG16
* Patch Sampling: patchwise training is loss sampling<br>
filter size로 나누지 않고 full images를 인풋으로 사용한다.
* Network can work regardless of the original image size, without requiring any fixed number of units at any stage

# 3. Skip Architecture
<img src='https://i.imgur.com/arpC1fB.png' width='100%'>
<img src='https://i.imgur.com/3IF0vjC.png' width='100%'>
> CAE(Convolutional Auto Encoder)와는 달리 FCN은 spatial dimension을 유지하는 것이 중요하다. 하지만 downsampling을 할수록, 그 정보가 손실되기 마련이다. 따라서 이러한 정보 손실을 막기 위해 FCN은 'skip connection'을 이용하여 기존의 위치 정보 및 pixel들간의 관계를 유지시켜준다.<br>
To fully recover the fine-grained spatial information lost in the pooling or downsampling layers, we often use skip connections

# 4. Architecture
<img src='https://i.imgur.com/71WUEsP.png' width='100%'>
> binary segmentation --> output channel = 2

# 5. Results
* 평가지표: pixel acc, mean acc, meanIU
* MIoU(Mean Intersection over Union)
    * 여러 장의 이미지에 대해 성능을 평가하기 위해 한 장마다의 IoU값을 평균내어 사용
    * 일반적으로 IoU의 threshold값으로 0.5를 잡아서 사용(=AP50)

<img src='https://i.imgur.com/Wr9mwVJ.png' width='100%'>
# 6. Reference
https://www.youtube.com/watch?v=2f89FOKu_kU