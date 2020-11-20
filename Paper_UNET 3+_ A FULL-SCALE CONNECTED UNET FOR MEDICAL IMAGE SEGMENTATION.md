# UNET 3+: A FULL-SCALE CONNECTED UNET FOR MEDICAL IMAGE SEGMENTATION

---
Paper) https://arxiv.org/abs/2004.08790<br>
Github) https://github.com/ZJUGiveLab/UNet-Version<br>
Refer) https://whereisend.tistory.com/214?category=849932<br>
<br>

# Abstract

*UNet++ was developed as a modified Unet by designing an architecture with nested and dense skip connections. However, it does not explore sufficient information from full scales and there is still a large room for improvement.*<br>
U-Net을 사용하며 모델의 성능을 올리기 위해 유의미한 backbone을 survey하던 중에 U-Net++라는 모델이 있다는 것을 확인했다. 모델의 성능 향상을 기대하며 U-Net++ inference 준비를 하고 있었는데, U-Net 3+라는 모델이 20년 4월에 발표되었다는 것도 확인했다. U-Net++는 물론이며, 아직 U-Net을 논문으로는 접해보지 못했지만, U-Net 3+ 논문을 읽으면서 해당 모델을 최대한 이해해보고자 한다. 이전 버전의 내용에 대하여 이해가 되지 않는 부분은 시간상의 문제로 빠르게 survey하면서 개념을 익혀나갈 것이지만, 그래도 이해가 되지 않으면 스킵하도록 하겠다. 기존 버전 모델들의 성능도 좋았지만 해당 논문에서는 기존 모델들이 전체 스케일에 대한 정보를 충분히 탐색하지 못하기 때문에 아직 성능 향상의 가능성이 있다고 밝히고 있다.<br>
<br>

*In this paper, we propose a novel UNet 3+, which takes advantage of full-scale skip connections and deep supervisions.*<br>
U-Net 3+는 크게 두 가지 특징을 가지고 있다. 하나는 full-scale skip connection이고, 다른 하나는 deep supervision이다. 두 특징을 기반으로 모델을 이해할 필요가 있다.<br>
<br>

*The full-scale skip connections incorporate low-level details with high-level semantics from feature maps in different scales; while the deep supervision learns hierarchical representations from the full-scale aggregated feature maps.*<br>
위에서 언급한 두 가지 특징에 대하여 간략하게 소개한다. full-scale skip connecetion은 서로 다른 scale에서 저수준과 고수준의 feature map을 통합해주는 skip connection이 존재하는 것으로 보인다. deep supervision은 계층적 표현을 배우는 역할을 하는 것으로 보인다. 아직 정확하게 특징을 이해하기가 어렵다.<br>
<br>

*The proposed method is especially benefiting for organs that appear at varying scales.*<br>
특히 해당 모델은 다양한 스케일을 가지고 있는 대상에 대해 성능이 뛰어나다고 말한다.<br>
<br>

*In addition to accuracy improvements, the proposed UNet 3+ can reduce the network parameters to improve the computation efficiency.*<br>
정확도 향상, 파라미터 수 감소로 인한 계산 효율성 향상 등의 장점도 가지고 있다.<br>
<br>

*We further propose a hybrid loss function and devise a classification-guided module to enhance the organ boundary and reduce the over-segmentation in a non-organ image, yielding more accurate segmentation results.*<br>
hybrid loss function과 classification-guided module 또한 제안하며 모델의 성능을 향상시키는 기법들을 소개할 것임을 보여준다.<br>
<br>

# 1. Introduction

*It uses skip connections to combine the high-level semantic feature maps from the decoder and corresponding low-level detailed feature maps from the encoder.*<br>
논문에서는 먼저 U-Net과 U-Net++의 특성 및 한계점에 대하여 언급한다. U-Net은 skip connection을 사용하여 encoder의 저수준 FM(feature map)을 decoder의 대응되는 고수준 FM과 조합하는 구조이다(skip connection의 효과는 사이즈가 작아진 FM을 decoding하면서 부족해진 정보를 encoder의 FM을 통해 보완하는 정도로 이해하고 있는데, 좀 더 확실한 이해 후에 수정하도록 하겠다).<br>
<br>

*To recede the fusion of semantically dissimilar feature from plain skip connections in UNet, UNet++ [7] further strengthened these connections by introducing nested and dense skip connections, aiming at reducing the semantic gap between the encoder and decoder.*<br>
U-Net은 단순히 encoder와 decoder의 대응되는 구조로 skip connection이 이루어졌기 때문에 유사하지 않은 FM의 혼합이 문제가 된다고 한다(이해가 부족한 부분으로, U-Net++ paper를 참고해야 할 필요가 있다). 따라서 U-Net++는 nested and dense skip connection(중첩되고 밀도있는 컨셉으로 판단된다)을 제안하여 encoder와 decoder 사이의 gap을 줄였다고 한다.<br>
<br>

*Despite achieving good performance, this type of approach is still incapable of exploring sufficient information from full scales.*<br>
하지만 이러한 시도에도 불구하고 여전히 full scale에 대하여 충분히 탐색하지 못하는 문제점을 가지고 있다고 지적한다. 이러한 기존 버전의 모델들이 문제점을 가지고 있기 때문에 U-Net 3+가 발표되었고, 이는 full scale 탐색이 가능하다는 특징을 가지고 있을 것으로 예상할 수 있다.<br>
<br>

*To make full use of the multi-scale features, we propose a novel U-shape-based architecture, named UNet 3+, in which we re-design the inter-connection between the encoder and the decoder as well as the intra-connection between the decoders to capture fine-grained details and coarsegrained semantics from full scales.*<br>
U-Net 3+은 기존 버전 모델들의 문제점을 해결하여 multi-scale feature들을 모두 탐색할 수 있다는 특징을 가진다. 이를 위해 encoder와 decoder 사이의 inter-connection, 그리고 decoder끼리의 intra-connection을 사용하여 모델을 새롭게 디자인했다.<br>
<br>

*To further learn hierarchical representations from the full-scale aggregated feature maps, each side output is connected with a hybrid loss function, which contributes to accurate segmentation especially for organs that appear at varying scales in the medical image volume.*<br>
또한 full-scale에서 가져온 정보를 계층적으로 학습하기 위해 각각 side output에 hybrid loss function을 적용한다. side output에 loss function을 적용한다는 의미는 최종 output뿐만 아니라 각 side output에서도 학습을 수행한다는 의미이기도 하다(해당 내용에 의해 아마 모델을 training할 때 각 scale에 대한 output이 준비되어야 하지 않을까 하는 추측을 했다). 이를 통해 다양한 크기를 가지고 있는 물체의 탐지력을 높여주는 효과를 가져온다.<br>
<br>

*Different from these methods, we extend a classification task to predict the input image whether has organ, providing a guidance to the segmentation task.*<br>
medical image에서 segmentation 정확도를 높이기 위해, 저자는 non-object image에서 FP(False Positive)를 효율적으로 줄일 수 있는 방법을 고민했다. 제시한 방법은, image를 U-Net 3+에 입력하기 전에 object가 존재하는지를 classification하는 과정을 추가하는 것이다.<br>
<br>

*Introduction Summary*
* 모든 scale에서 low-level과 high-level을 통합하는 full-scale skip connection을 사용함으로써, multi-scale feature를 모두 사용한다.
* hybrid loss function을 통해 계층적 표현을 학습할 수 있도록 deep supervision을 수행한다.
* FP를 줄이기 위해 classification-guided module을 적용한다.
* 광범위한(다양한 조합의?) experiment를 수행한다.
<br>

*discussion*<br>
해당 논문에서는 *fine-grained*와 *coarse-grained*라는 용어가 자주 등장한다. 의미로 추론해봤을 때 coarse-grained는 좀 더 넓은 범위에서의 task를 의미하고, 반대로 fine-grained는 세부적인 범위에서의 task를 의미한다. computer science분야에서 사용되는 두 용어의 의미를 정리해둔 자료가 있어서 reference와 함께 따로 정리해두겠다.<br>

* 병렬 컴퓨팅에서는 프로세서로 데이터가 가는 횟수와 양에 따라서 쓰인다고도 한다. fine-grained한 데이터 전송은 적은 양의 데이터를 자주 보내는 것을 의미하고 coarse-grained는 반대이다.
* Reconfigurable Computing에서는 데이터 흐름의 크기에 따라서도 이 용어가 쓰인다. FPGA와 같은 회로작업에서는 데이터 흐름의 단위가 단일 비트인데 이 경우 fine-grained computing이라고 한다. 반대로 coarse-grained computing은 데이터 흐름의 단위가 32bit와 같이 상대적으로 크기가 클 경우에 쓰인다.
* Data granularity에서는 데이터가 얼마나 자세히 분할되어있는지 여부에 따라서 fine 혹은 coarse grained가 쓰인다. 주소라는 구조체를 정의할 때 단일 필드에 모든 내용을 쓰게 되면 상대적으로 coarse-grained가 되고, 여러 필드로 나누어(지역, 동, 아파트 이름) 쓰게 되면 상대적으로 fine-grained가 된다.

Reference) https://lastyouth.tistory.com/4<br>
<br>

# 2. Method

<img src='https://i.imgur.com/rO2dRWF.png' width='100%'>

## 2.1 Full-scale Skip Connections

*To remedy the defect in UNet and UNet++, each decoder layer in UNet 3+ incorporates both smaller- and same-scale feature maps from encoder and larger-scale feature maps from decoder, which capturing fine-grained details and coarse-grained semantics in full scales.*<br>
U-Net의 plain connection과 U-Net++의 nested and dense connection의 불충분한 정보의 탐지로 인해, U-Net 3+는 다른 방법으로 skip-connection을 구상했다. 각 decoder layer는 자신과 같거나 더 작은 encoder FM과 통합하고 자신보다 더 큰 decoder FM과 통합한다. 이 방법으로 모든 scale에서 fine-grained details와 coarse-grained semantics를 얻을 수 있다고 한다.<br>
<br>

> How to construct the feature map of decoder layer<br>
* same-scale encoder: directly received in the decoder(similar to the U-Net)
* smaller-scale encoder: by applying non-overlapping max pooling operation
* larger-scale decoder: by utilizing bilinear interpolation

<img src='https://i.imgur.com/3ffGEmk.png' width='100%'>
*To seamlessly merge the shallow exquisite information with deep semantic information, we further perform a feature aggregation mechanism on the concatenated feature map from five scales, which consists of 320 filters of size 3 × 3, a batch normalization and a ReLU activation function.*<br>
skip-connection이 수행된 이후 low level의 정보와 high level의 정보를 부드럽게 이어주기 위해 filter=320인 3x3 convolution, BN, ReLU를 수행한다. 그리고 논문에서는 해당 메커니즘을 feature aggregation mechanism이라고 통칭하고 있다.<br>
<br>

*It is worth mentioning that our proposed UNet 3+ is more
efficient with fewer parameters.*<br>
U-Net 3+의 또 다른 장점은 기존 버전의 모델들보다 더 적은 파라미터를 가지고 있다는 것이다.<br>
<br>

## 2.2 Full-scale Deep Supervision
*In order to learn hierarchical representations from the fullscale aggregated feature maps, the full-scale deep supervision is further adopted in the UNet 3+.*<br>
Introduction에서 설명한 U-Net 3+의 대표 특징 중 하나인 full-scale deep supervision이다. 이는 모든 스케일의 feature map으로부터의 계층적 정보를 학습하기 위해 고안되었다.<br>
<br>

*the proposed UNet 3+ yields a side output from each decoder stage, which is supervised by the ground truth. To realize deep supervision, the last layer of each decoder stage is fed into a plain 3 × 3 convolution layer followed by a bilinear up-sampling and a sigmoid function.*<br>
full-scale deep supervision은 각 decoder stage에서 수행된다. ground truth와 비교하기 위해 각 decoder stage는 3x3 conv --> bilinear up-sampling --> sigmoid 프로세스를 거쳐 스케일을 맞춰준다(sigmoid functioin을 사용한다고 했는데, 그럼 multiclass segmentation은 어떻게 수행하는지가 의문이다).<br>
<br>

<img src='https://i.imgur.com/tCtbN3M.png' width='100%'>

* Focal Loss: 문제가 있는 loss에 더 가중치를 준다.
* MS-SSIM: 결과와 원래 이미지 간 밝기, 대조, 구조 값을 비교하는 ssim을 여러 스케일에서 점수를 내어 가중곱하는 형태
* IoU loss: 1 - IoU값을 loss로 사용함

*By combining focal loss, MS-SSIM loss and IoU loss, we develop a hybrid loss for segmentation in three-level hierarchy- pixel-, patch-, and map-level, which is able to capture both large-scale and fine structures with clear boundaries.*<br>
U-Net 3+는 위 3개의 loss function을 모두 합침으로써 pixel별, patch별, map-level별로 모델의 성능을 평가할 수 있도록 했다. 참고로 MS-SSIM은 탐지하고자 하는 대상의 경계(boundary)에 대한 검출력을 높이기 위해 설계된 loss function이라고만 언급하고 넘어가겠다.<br>
<br>

## 2.3 Classification-guided Module(CGM)
<img src='https://i.imgur.com/DGbdJMm.png' width='100%'>
*To achieve more accurate segmentation, we attempt to solve this problem by adding an extra classification task, which is designed for predicting the input image whether has organ or not.*<br>
medical image segmentation 분야는 FP가 피할 수 없는 문제점 중에 하나라고 언급한다. 예를 들어, 장기(organ)이 아닌데 검출하는 경우가 종종 있다는 의미가 된다. 이는 환자의 생명과 연관되어있는 의료분야에서는 치명적인 사안이 아닐 수 없다. 논문에서는 이에 대한 원인으로 얕은 layer에 남아 있는 noise information 때문에 과도하게 segmentation하는 현상이 발생한다고 언급하고 있다.U-Net 3+는 이 문제를 해결하기 위해 이미지 내에 장기가 있는지 없는지를 판단하는 classification task를 추가했다.<br>
<br>

해당 모듈의 작동 원리는 다음과 같다. CGM은 Dropout, Convolution, Maxpooling, Sigmoid 연산으로 이루어져 있다. 가장 깊은 level인 X5(En)(=X5(De))이 CGM 연산을 수행하고 나면 2d tensor 결과를 얻을 수 있다. 해당 tensor의 각 차원은 object의 with/without을 의미한다. 이후 크게 2단계를 거쳐 모델의 성능 향상에 도움을 준다고 볼 수 있다. 먼저 argmax function을 통과하여 2-d tensor는 0 또는 1의 단일 output을 출력한다. 두 번째로 segmentation의 side output과 이 값을 곱하여 다음 layer에 전달하게 되는 방식이다. 저자는 이를 통해 non-organ image에서 over-segmentation을 극복했다고 언급한다.<br>
<br>

# 3. Experiments and results
## 3.1 Dataset and Implementation
(...skip...)

## 3.2 Comparison with UNet and UNet++
(...skip...)

## 3.3 Comparison with the State of the Art
(...skip...)

# 4. Conclusions
* full-scale skip connection --> segmentation 정확도 향상
* deep supervision
* fewer parameters
* classification-guided module(CGM) --> 정확한 위치와 경계 인식 맵 생성
* hybrid loss function --> 정확한 위치와 경계 인식 맵 생성