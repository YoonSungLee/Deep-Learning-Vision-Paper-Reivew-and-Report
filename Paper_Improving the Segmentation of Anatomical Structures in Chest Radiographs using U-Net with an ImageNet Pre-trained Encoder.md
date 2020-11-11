# Improving the Segmentation of Anatomical Structures in Chest Radiographs using U-Net with an ImageNet Pre-trained Encoder

---

Reference) https://www.groundai.com/project/improving-the-segmentation-of-anatomical-structures-in-chest-radiographs-using-u-net-with-an-imagenet-pre-trained-encoder/1<br>
<br>

# Abstract
*In this paper we investigate the latest fully-convolutional architectures for the task of multi-class segmentation of the lungs field, heart and clavicles in a chest radiograph. In addition, we explore the influence of using different loss functions in the training process of a neural network for semantic segmentation.*<br>
해당 논문은 fully-convolutional architecture를 연구하는 내용으로써, pre-trained encoder weights(VGG16)을 backbone으로 하는 U-net을 대상으로 여러 loss function을 사용하여 모델의 성능을 올린 experiment를 소개한다.<br>
<br>
# 1. Introduction
*The training process of each CNN is affected by several training features.*<br>
chest radioograph에 대한 도메인을 소개하며, 해당 분야에 대한 task를 수행하는 방법론중의 하나로 CNN 모델을 언급한다. 특히 CNN 모델을 훈련하는 중에 많은 영향을 주는 요인이 존재하는데, 해당 논문에서는 loss function과 weights initializer(random initialization or weights transferred from another trained network)를 언급한다.<br>
<br>

*We propose an improved encoder-decoder style CNN with pre-trained weights of the encoder network and show its superiority over other state of the art CNN architectures.*<br>
논문에서는 pre-trained weights를 가진 encoder-decoder 형식의 CNN 모델을 소개하며, 해당 모델의 우수한 성능을 검증하는 과정을 소개한다.<br>
<br>

*We further examine the use of multiple loss functions for training the best selected network and the effect of multi-class vs. single-class training.*<br>
또한 다수의 loss function을 적용하고 성능을 비교한 experiment를 소개한다.<br>
<br>

# 2. Method
## 2.1 Fully Convolutional Neural Network Architectures
*Iglovikov et al [17] proposed to use a VGG11 [12] as an encoder which was pre-trained on ImageNet [18] dataset and showed that it can improve the standard U-Net performance in binary segmentation of buildings in aerial images.*<br>
U-Net의 성능을 향상시키기 위해 수행되었던 연구 중에 하나를 소개한다. 이는 VGG11을 encoder로(즉 backbone으로) 사용하여 binary segmentation의 성능을 향상시켰다는 내용이다. 이처럼 여러 구조의 모델을 U-Net의 backbone으로 사용하여 모델의 성능을 향상시키는 연구가 진행중이다. 이는 backbone의 역할이 feature extraction을 수행하는 본래의 목적을 가지고 있기 때문에, 이를 잘 수행하는 구조를 사용하면 그만큼 성능이 올라가기 때문이라고 짐작할 수 있다.<br>
<br>

*A similar concept was used in the current study with the more advanced VGG16 [12] as an encoder.*<br>
위와 유사한 연구 중의 하나로써, VGG16을 backbone으로 사용한 U-Net이 있다. 논문의 저자는 이 구조를 사용하여 수행한 실험을 소개한다. 구조는 다음과 같다.<br>

<img src='https://i.imgur.com/XYTopBF.png' width='100%'>

## 2.2 Objective loss functions
<img src='https://i.imgur.com/VlfT2gL.png' width='100%'>

*The Dice similarity coefficient (DSC) and Jaccard similarity coefficient (JSC) are two well known measures in segmentation and can be used as objective loss functions in training.*<br>
<br>

<img src='https://i.imgur.com/liUlnPJ.png' width='100%'>

*The Tversky loss [19] introduces weighting into the loss function for highly imbalanced data, where we want to segment small objects. where α and β control the magnitude of penalties for FPs and FNs, respectively. In our study we used α = 0.3 and β = 0.7.*<br>
<br>

<img src='https://i.imgur.com/ESsaOnd.png' width='100%'>

*BCE was calculated separately for each class segmentation map.*<br>
<br>

# 3. Segmentation of Anatomical Structures
* 4가지 모델을 실험했는데, U-Net with VGG16의 성능이 가장 뛰어났다.
* 밑바닥부터 학습하는 것보다 pre-trained weights를 학습하는 것이 성능이 더 뛰어났다.
* 쇄골과 같이 픽셀 면적이 작은 object는 (Tversky loss function과 같이) 픽셀에 가중치를 부여하는 loss function을 통해 학습이 잘 이루어졌다. 그 반대의 경우는 Dice loss 또는 Binary Cross-Entropy loss를 사용하는 경우 학습이 잘 이루어졌다.
<br>

# 4. Discussion and Conclusion
(...skip...)
