# Image Instance Segmentation with Mask R-CNN

---

# Abstract

Deep Learning 기술을 Computer Vision분야에 적용함으로써 많은 전문적인 분야들이 파생되었다. classification, segmentation, object detection, GAN 등이 이에 해당하는데, 특히 segmentation과 object detection을 접목해서 성능을 높인 ‘Mask R-CNN’ 모델을 깊이 연구하기로 한다. 구체적인 사항으로는 Mask R-CNN의 논문을 리뷰하고 토의하는 방식을 통해 모델을 이해한다. 그리고 Keras로 구현된 오픈소스를 활용하여 모델을 사용하고 성능을 평가한다.

# Object Detection

<img src="https://i.imgur.com/dXc6g6c.png" width="100%">

Mask R-CNN을 이해하기 위해서는 그 기저에 깔린 Object Detection과 Segmentation이 무엇인지 먼저 알 필요가 있다. 위 그림을 살펴보자.<br>
먼저 Semantic Segmentation의 경우, 각각의 픽셀에 대하여 해당 픽셀이 어떤 클래스에 속하는지를 분류한다. 위 그림의 경우 각 픽셀이 GRASS, CAT, TREE, SKY 중에 어떤 것에 해당하는지를 분류하고 값을 다르게 매긴 것을 확인할 수 있다.<br>
Classification 그리고 Localization의 경우, Classification은 단순히 사진 속의 객체가 어떤 클래스를 의미하는지를 분류하고, Localization은 그 객체의 Bounding Box를 추출해낸다. 이 때 Classification은 분류 문제에 해당하고, Localization은 회귀 문제에 해당한다. 단일 객체만을 분류 및 Bounding Box를 할 수 있다는 것이 특징이다.<br>
Object Detection의 경우, 여러 객체에 대하여 Classification과 Bounding Box를 추출해낸다. 특히 위의 그림처럼 같은 클래스(DOG)라도 각자 Classification 및 Bounding Box를 추출해낸다는 것이 특징이다. 따라서 앞의 문제에 비해 고난이도의 문제라고 할 수 있다.<br>
마지막으로 Instance Segmentation의 경우, 모든 객체(마찬가지로 같은 클래스라도 각자 수행)에 대하여 Classification 및 Bounding Box를 진행하고, 여기서 더 나아가 각 Bounding Box내에 Segmentation을 수행한다. 따라서 위의 그림처럼 클래스를 넘어서 모든 객체를 Classification하고 Segmentation한 결과를 확인할 수 있다. 우리가 살펴볼 Mask R-CNN은 이에 해당한다.

Mask R-CNN을 잘 이해하기 위해서는 결국 그 기저에 깔린 수많은 내용들을 이해해야 할 것이다. 이 모델은 갑자기 튀어나온 것이 아니라 사실 Object Detection을 위해 만들어진 모델에서 출발했다. 따라서 R-CNN 시리즈 모델이라고도 할 수 있는, R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN의 주요 특징들과 변천과정을 이해한다면 Mask R-CNN을 더욱 잘 이해할 수 있을 것이다.

# R-CNN(2013)

### Model

<img src="https://i.imgur.com/HIpL2ks.png" width="100%">

<img src="https://i.imgur.com/jrU53ke.png" width="100%">

R-CNN의 작동원리와 구조는 위 두 그림과 같다.<br>
먼저 Input Image에서 약 2000여장의 이미지를 'Selective Search' 알고리즘을 통해 추출한다. 이 알고리즘에 대한 자세한 내용은 아래에서 설명한다. 이 이미지는 우리가 관심있어하는 정보이기에 'RoI(Region of Interest)'라고 부른다. 추출한 RoI를 CNN 모델에 입력하기 위해 Input Size를 동일하게 맞춰준다. 이 과정을 'Warp'이라고 한다. 그리고 CNN 모델은 2개의 연산으로 나뉘게 되는데, 하나는 분류를 위한 Classification이고, 나머지 하나는 Bounding Box를 정교하게 추출하기 위한 Regression이다. Bounding Box Regression은 Neural Network로 구성되어 있는 반면, Classification은 SVM이라는 다른 모델을 사용한다. 따라서 CNN 모델의 마지막 feature map을 SVM에 적용하여 Classification을 수행한다.

### Selective Search

<img src="https://i.imgur.com/V2dnlCS.png" width="100%">

Selective Search란 Object Detection에서 입력된 이미지의 후보 영역을 추천할 때 사용되었던 방법으로 R-CNN 등의 논문들이 탁월한 성능을 보여주면서 주목받게 되었다. 특히 이는 2가지 방법을 조합하여 수행하는데, Exhaustive Search와 Segmentation이 이에 해당한다. 전자의 경우 모든 가능한 객체의 위치를 찾아내는 것을 의미하고, 후자의 경우 이미지의 구조적 특징(색상, 무늬, 크기, 모양)을 사용하여 후보 영역을 추출하는 것을 의미한다.

<img src="https://i.imgur.com/71hypJX.png" width="100%">

알고리즘은 다음과 같은 방식으로 수행된다.<br><br>
1. Efficient Graph-Based Image Segmentation(Felzenszwalb)을 사용하여 초기 후보 영역을 다양한 크기와 비율로 생성한다.
2. 그리디 알고리즘을 통해 비슷한 영역을 반복적으로 통합한다.
    * 처음에 모든 영역에 대해 유사도를 계산하여 similarity set S를 생성한다.
    * S에서 가장 큰 유사도 값을 가진 r(i), r(j)에 대해 통합한다.
    * r(i), r(j)의 유사도 값은 S로부터 제거한다.
    * 통합된 새로운 영역(r(t))과 인접한 영역들에 대해 유사도(S(t))를 계산한다.
    * S와 R에 유사도(S(t))와 통합된 새로운 영역(r(t))을 추가한다.
3. 최종적으로 하나의 영역이 만들어질 때까지 2번을 반복한다.

### R-CNN Training

<img src="https://i.imgur.com/6yOt47e.png" width="100%">

* Pre-train a ConvNet(AlexNet) for ImageNet classification dataset
* Fine-tune for object detection(softmax + log loss)
* Cache feature vectors to disk
* Train post hoc linear SVMs(hinge loss)
* Train post hoc linear bounding-box regressors(squared loss)

모델 구성을 조금 더 설명하자면 ImageNet classification dataset으로 Pre-train된 모델을(위의 그림에서는 AlexNet을 예로 들고 있다)을 ConvNet으로 활용한다. 이 때 AlexNet의 마지막 layer를 Obejct Detection을 위한 class의 수로 수정하여 fine tuning을 수행한다. 이후 마지막 layer의 이전 layer를 디스크에 따로 저장을 해놓는다. 그리고 그 저장된 데이터를 SVM에 학습시킨다.<br><br>
* fine tuning: 기존에 학습되어져있는 모델을 기반으로 아키텍처를 새로운 목적(나의 이미지 데이터에 맞게)에 맞게 변형하고 이미 학습된 모델 weights로부터 학습을 업데이트하는 방법을 말한다. 이는 모델의 파라미터를 미세하게 조정하는 행위로써 특히 딥러닝에서는 이미 존재하는 모델에 추가 데이터를 투입하여 파라미터를 업데이트하는 것을 말한다.

Question) Bounding-box regressor는 어떤 layer를 통해 학습하는가? 그리고 이 과정은 end-to-end인가?(PPT 12쪽에 따르면 update되지 않는다고 쓰여져 있음)<br><br>
Answer) BBox regression은 분리된 Neural Network로써 ConvNet과 함께 가중치들이 update되지 않는다. 따라서 end-to-end 학습이라고 할 수 없다. 즉, Selective Search, ConvNet, SVM, BBox Regerssion 모두가 각자의 모델로 존재한다.

### Bounding-Box Regression

<img src="https://i.imgur.com/5ikzHbY.png" width="100%">

<img src="https://i.imgur.com/dKxKXMX.png" width="100%">

<img src="https://i.imgur.com/zjXZJ8Q.png" width="100%">

Bounding-Box Regression의 작동원리는 Bounding Box의 중심 x좌표 x, y좌표 y, 높이 h, 너비 w를 예측한 P와 실제값 G를 줄이는 방식으로 진행된다. 특히 P와 G의 차이를 나타내는 t와 우리가 조절해야 할 대상인 d(P)의 간격을 최소화하는 방향으로 이루어진다.

Question) Bounding-Box Regression 위의 내용이 맞는지 여부?<br><br>
Answer) 추후 확인 필요

### Problems of R-CNN

R-CNN의 문제점은 다음과 같다.<br><br>
1. Slow at test-time<br>
R-CNN의 구조를 보면 처음 Selective Search를 통해 얻은 2000여장의 Region Proposal을 ConvNet에 모두 입력시켜야한다. 이는 test시에 굉장히 많은 시간을 필요로 하게 된다. 특히 논문에 게재된 내용에 따르면 test시 image 한 장당 GPU(K40)를 이용하여 13초의 시간을, CPU를 이용하여 53초의 시간을 필요로 했다.
2. SVM and regressors are post-hoc<br>
ConvNet과 SVM, Regressor는 별개의 모델로써 end-to-end 학습을 수행할 수 없다. 따라서 SVM과 Regressor에서 학습한 내용을 ConvNet에 전달할 수 없다.
3. Complex multistage training pipeline(84 hours using K40 GPU)<br>
R-CNN의 구조는 multistage training pipeline으로써 많은 모델을 한꺼번에 가지고 있다. ConvNet, SVMs, Bounding-Box Regressor가 이에 해당하는데, 이들을 모두 학습시켜야하기때문에 굉장히 복잡한 구조를 가지고 있다고 할 수 있다.

# Fast R-CNN(2015)

위의 R-CNN의 문제점들을 보완해서 나온 모델이 Fast R-CNN이다. 아래 그림을 통해 자세히 살펴보자.

### Model

<img src="https://i.imgur.com/gM5oGFt.png" width="100%">

<img src="https://i.imgur.com/m31U15k.png" width="100%">

Selective Search를 통해 RoI를 추출하는 것은 동일하다. 하지만 R-CNN과는 달리 이미지 전체를 ConvNet에 바로 입력시킨다. 이 때 Input Image가 ConvNet을 거치면서 사이즈가 pooling되는데 RoI 또한 같이 pooling된다고 볼 수 있다. 이를 RoI projection이라고 한다. 이후 각각의 Bounding Box를 pooling하여 같은 사이즈로 맞춰주는 RoI pooling을 수행한다. 이는 Fully Connected Layer에 동일한 사이즈로 입력시켜주기 위함이다. RoI pooling으로 인해 모두 동일한 사이즈를 갖게 된 데이터는 FC를 거쳐 RoI feature vector가 되고, 또 다시 병렬적인 FC를 거쳐 Softmax(classification)와 Bounding Box Regression을 수행하는 구조로 이루어져있다. 즉 Fast R-CNN의 핵심은 RoI pooling이라고 할 수 있겠다.

### RoI Pooling

<img src="https://i.imgur.com/BV8xO56.png" width="100%">

<img src="https://i.imgur.com/zQww6ym.png" width="100%">

ConvNet을 통과한 feature map에 존재하는 Bounding Box를 다음 FC에 입력하기 위해 모두 같은 사이즈로 만들어주는 것을 RoI Pooling이라고 했다. 구체적으로 어떻게 작동하는지 살펴보자.<br>
예를 들어 FC에 입력할 사이즈를 7x7로 정해놓았다고 가졍하자. 그럼 ConvNet을 통과한 feature map에 존재하는 모든 RoI의 사이즈를 7x7로 만들어주어야한다. 어떤 RoI의 사이즈가 21x14라고 하면, (3,2)의 stride를 가진 3x2 max pooling을 통해 7x7로 만들어줄 수 있다. 마찬가지로 또 다른 RoI의 사이즈가 35x42라고 하면, (5,6)의 stride를 가진 5x6 max pooling을 통해 7x7로 만들어줄 수 있다. 이처럼 모든 RoI에 각각에 맞는 stride를 적용하여 max pooling을 한다면 모두 같은 사이즈로 만들어 줄 수 있다. 이를 보기 편하게 설명하기 위해 아래와 같이 정리할 수 있다.<br><br>
RoI in Conv feature map : 21x14 -> 3x2 max pooling with stride(3, 2) -> output : 7x7<br>
RoI in Conv feature map : 35x42 -> 5x6 max pooling with stride(5, 6) -> output : 7x7

### Training & Testing

<img src="https://i.imgur.com/vNfzMVF.png" width="100%">

<img src="https://i.imgur.com/DatXybA.png" width="100%">

Fast R-CNN을 훈련시킴에 있어서 Loss Function을 눈여겨 볼 만하다. L(cls)는 우리가 흔히 알고 있는 Cross Entropy를 사용했고, L(loc)은 새롭게 함수를 정의해놓았다. 특히 u가 1 이상일 경우를 설정해놓았는데, 이는 u가 0인 경우, 즉 배경인 경우에는 L(loc)을 고려하지 않겠다는 것을 의미한다. 이는 엉뚱한 곳에 RoI를 잡은 경우 Bounding Box Regression에 대한 loss는 구하지 않도록 해준다. 특히 IoU에 대한 임계값을 설정하면 그 임계값 이상인 RoI에 대해서만 loss를 구하게 된다.

### R-CNN vs. SPP-net vs. Fast R-CNN

<img src="https://i.imgur.com/9KEPmyF.png" width="100%">

그렇다면 Fast R-CNN은 얼마나 성능이 향상되었을까? 위의 그래프를 통해 확인할 수 있다. 기존 R-CNN에 비해 Training time과 Test time을 굉장히 단축시킨 것을 확인할 수 있다. 특히 눈여겨 봐야 할 점은 오른쪽 그래프의 Fast R-CNN이다. 파란색 그래프는 Region Proposal를 포함시킨 시간이고, 빨간색 그래프는 포함시키지 않은 시간이다. 즉, 이 두 시간의 차이는 Region Proposal을 구하는 시간인 Selective Search를 의미한다. 이 시간이 전체 시간의 대부분을 차지하고 있음을 확인할 수 있다. 이는 앞에서도 언급했다시피 CPU상에서 작동하기 때문이다.

### Problems of Fast R-CNN

Fast R-CNN의 문제점은 다음과 같다.<br><br>
1. Out-of-network region proposals are the test-time computational bottleneck<br>
Region Proposal, 즉 Selective Search가 Network의 바깥에 존재하여 시간이 굉장히 오래 걸린다.
2. Is it fast enough?<br>
과연 Fast R-CNN의 성능을 빠르다고 할 수 있는가는 생각해봐야한다. Real-Time을 적용하기 위해서는 2.3초는 빠르다고 할 수는 없을것이다.

# Faster R-CNN(2015)

### Model

R-CNN, Fast R-CNN은 CPU에서 수행하는 Selective Search때문에 많은 시간을 필요로 했다. 이를 마지막 convolution layer 이후에 RPN(Region Proposal Network)을 사용하여 GPU를 사용해보자는 것이 Faster R-CNN의 아이디어이다. 따라서 RPN은 Region Proposal을 정확하게 잡아내도록 학습되고, RPN 이후 RoI Pooling을 하여 Classifier와 BBox regressor를 사용하는 것은 Fast R-CNN과 동일하다. 즉, Faster R-CNN을 한마디로 표현하면 RPN + Fast R-CNN이라고 할 수 있다.

<img src="https://i.imgur.com/d33QiVJ.png" width="100%">

Question) 기존 Fast R-CNN의 Selective Search가 느려서 RPN을 사용하여 Faster R-CNN이 나온 것으로 알고 있습니다. 이 때 RPN에 있는 reg가 BBox를 조정해주는 것으로 알고 있는데, 그럼 원래 Fast R-CNN의 reg는 없어져도 되지 않나요? reg가 2개가 있는 이유가 궁금합니다.<br><br>
Answer) ROI pooling 된 결과가 CNN에 입력으로 들어가야 하는데 CNN의 입력은 고정된 사이즈를 요구합니다. 따라서 정확한 ROI위치를 잡아서 CNN의 입력으로 넣어주기 위해서 첫번째 BB regression이 필요한것이구요. 두번째는 이제 object dection된 결과를 평가함에 있어 IOU와 같이 BB overlap이 얼마나 잘 됐는지를 평가하는 metirc이 있기 때문에 여기서 성능이 낮아지는걸 방지하기 위해 Refine BB 가 있는것입니다.

Question) 정확한 ROI 위치를 잡기 위해 BBox를 조정하는 것이 곧 BB overlap이 높다는 것을 의미하는 것 아닌가요?<br><br>
Answer) 그렇긴한데요. 결국 BB 오버랩은 맨 마지막에 측정하기 때문에 출력단에 refine BB가 있습니다.

Question) 그럼 마지막 reg는 BBox 미세조정도 하지만 측정을 위해서 존재의미가 더 큰건가요?<br><br>
Answer) 네네 BB 가 조금만 벗어나도 성능에 큰 영향을 미치기 때문에 사실 object를 detection하는데 있어서는 의미가 없지만 성능을 보여주기 위해서는 평가지표를 따라야 하기 때문에 평가지표상 성능을 높이기 위해서 존재한다고 보셔도 됩니다.

### RPN

<img src="https://i.imgur.com/AwZ6GyL.png" width="100%">

* 3x3 conv, 256 filters
* 1x1 conv, 18 filters
* 1x1 conv, 36 filters

그렇다면 RPN은 도대체 뭘까? 시작은 ConvNet을 통해 얻어진 feature map에서 시작한다. 여기서 동일한 크기의 sliding window를 이동시키며 window의 위치를 중심으로 사전에 정의된 다양한 비율/크기의 anchor box들을 적용하여 feature를 추출한다. 위의 그림에서는 ZF 기준으로 3x3 conv를 통해 256개의 feature를 추출하는 것을 확인할 수 있다. 이는 image 크기를 조정할 필요가 없으며, filter 크기를 변경할 필요도 없으므로 계산효율이 높은 방식이라 할 수 있다. 이후 추출된 feature를 1x1 conv를 수행하고 그 값을 바탕으로 병렬적으로 Classification과 BBox Regression을 수행하는 방식이라고 할 수 있겠다. 이 때 네트워크를 heavy하지 않고 slim하게 만들기 위해 물체가 존재하느냐 하지 않느냐에 따른 2개의 classification만을 수행한다.

<img src="https://i.imgur.com/UpSraSD.png" width="100%">

그렇다면 initial한 BBox가 필요할텐데, 그것을 'anchor'라고 한다. k개의 anchor box를 미리 정의해놓고 각 지역마다 사용하는 방식이다. 논문에서는 k개의 anchor box를 3가지 크기에 대한 3가지 비율을 경우의 수로 두고 9개의 anchor box로 정의했다. 이후 각각의 anchor box에 대해서 Classification과 BBox Regression을 수행하는 것이다. 위의 그림을 보면 Classification부분은 물체 존재여부에 따라 2개의 경우로 나뉘므로 총 2k개의 socres가 도출되고, BBox Regression부분은 x, y, h, w 4개의 값을 필요로 하므로 4k개의 coordinates가 도출된다.

* anchor: pre-defined reference boxes
* Multi-scale/size anchor: 3 scale(128x128, 256x256, 512x512) and 3 aspect rations(2:1, 1:1, 1:2) yield 9 anchors

### Positive/Negative Samples

* An anchor is labeled as positive if
  * The anchor is the one with highest IoU overlap with a ground truth box
  * The anchor has an IoU overlap with a ground truth box higher than 0.7
* Negative labels are assigned to anchors with IoU lower than 0.3 for all ground truth boxes
* 50%/50% ratio of positive/negative anchors in a minibatch

위의 설명과 같이 anchor box는 positive 또는 negative로 라벨링된다. ground-truth box와 가장 큰 IoU 값을 갖는 anchor와 0.7 이상의 값을 갖는 anchor는 positive로 라벨링된다. 반대로 IoU가 0.3 이하의 값을 갖는 anchor는 negative로 라벨링된다. 그 사이의 값들은 그냥 버리는 값이 된다. 특히 Random Sampling으로 학습을 하면 negative 레이블이 너무 많기 때문에 50대50 비율로 샘플링을 해준다.

### RPN Loss Function

<img src="https://i.imgur.com/7E1Yl8n.png" width="100%">

기존 Fast R-CNN의 Loss Function과 비슷한 Loss Function을 가진다. 특히 P_star는 물체가 있으면 1, 없으면 0의 값을 가지기 때문에 배경에 대해서는 Regression 계산을 하지 않는다. 추가적으로 논문에 따르면 N(cls)나 N(reg)처럼 Normalization하는 부분, 그리고 lambda는 크게 중요하지 않다고 말하고 있고 실험 결과를 통해 보여주고 있다.

Question) Loss Function 수식에 대한 이해?<br><br>
Answer) 수식에 대한 이해가 더욱 필요하다.

### 4-Step Alternating Training

<img src="https://i.imgur.com/cLc0heY.png" width="100%">

Question) 실제 Training에 대한 내용<br><br>
Answer) 이 부분은 실제 모델을 학습하고 Region Proposal을 출력하는 과정이다. 코드를 활용하기 위해 이해해야 할 부분인데 완벽히 이해하지를 못했다. 다른 자료들을 통해 이해해야 하는 과정이 필요하다.

### Results

<img src="https://i.imgur.com/In4IVHA.png" width="100%">

위의 표는 R-CNN, Fast R-CNN, Faster R-CNN의 정확도와 성능을 비교하고있다. 이미지 1장당 test time을 0.2초까지 줄였으며, 이는 R-CNN에 비해 250배 향상된 속도이다.

<img src="https://i.imgur.com/1Jmc0o3.png" width="100%">

또한 위의 표에서 VGG를 사용한 모델을 살펴보자. SS + Fast R-CNN은 Selective Search를 사용한 모델로써, 기존 Fast R-CNN을 의미한다. 그리고 그 아래 RPN + Fast R-CNN은 Faster R-CNN을 의미한다. 즉, CPU를 사용했던 Selective Search를 GPU를 사용하는 RPN으로 바꾸면 proposal부분에서 시간이 굉장히 단축됨을 확인할 수 있다.

### Problems of Faster R-CNN

Faster R-CNN의 문제점은 다음과 같다.
* RoI Pooling has some quantization operations
* These quantizations introduce misalignments between the RoI and the extracted features
* While this may not impact classification , it can make a negative effect on predicting bbox

Faster R-CNN 또한 마찬가지로 RPN에서 추출한 Region Proposal을 RoI Pooling을 통해 사이즈를 변환한다. 이 RoI Pooling 작동방식에서 위와 같은 문제가 발생한다. 이는 RoI Pooling을 사용하는 Fast R-CNN도 마찬가지이다. 위에서 RoI Pooling을 언급했던 예시를 다시 살펴보면 아래와 같다.<br><br>
RoI in Conv feature map : 21x14 -> 3x2 max pooling with stride(3, 2) -> output : 7x7<br>
RoI in Conv feature map : 35x42 -> 5x6 max pooling with stride(5, 6) -> output : 7x7<br><br>
이는 해당하는 두 개의 RoI가 7로 잘 나누어떨어지는 h와 w를 가지고 있는 경우인데, 예를 들어 세 번째 RoI가 21x15라면 어떤 방식으로 pooling을 해도 7x7의 출력을 할 수가 없다. 이는 Classification 부분에서는 큰 문제가 없을지 몰라도, BBox Regression에서 정보손실을 일으켜 부정적인 영향을 준다. 이해를 쉽게 하기 위해 아래 그림을 참조할 수 있겠다.

<img src="https://i.imgur.com/1KoOsmM.png" width="100%">

# Mask R-CNN

### Model

<img src="https://i.imgur.com/TDyfxMN.png" width="100%">

드디어 대망의 Mask R-CNN이다. 분위기상으로 보자면 이 모델은 이전 R-CNN 계열의 모든 문제점들을 해결한 것처럼 보이는데, 그게 맞다. 한 번 살펴보도록 하자.<br>
먼저 이 모델을 직관적으로 다음과 같이 이해할 수 있겠다. 우리가 해결하고자 하는 문제는 Instance Segmentation으로써, 모든 객체를 다르게 인식하고 segmentation 또한 할 수 있어야 한다. 어떻게 이 문제를 해결할 수 있을까? 단순하게 생각해보자면, 기존 Faster R-CNN은 BBox Classification을 아주 잘 해내는 모델이고 기존 FCN은 Segmentation을 아주 잘 해내는 모델이다. 그럼 이 두 모델의 장점만을 결합한다면? 이것이 바로 Mask R-CNN의 컨셉이다. 즉 Faster R-CNN이 검출해 준 BBox안에서 FCN을 통해 Segmentation을 하면 이 문제를 해결할 수 있다.

<img src="https://i.imgur.com/aZ1277L.png" width="100%">

### Mask Head

<img src="https://i.imgur.com/hMW3WMa.png" width="100%">

그럼 단순히 Faster R-CNN과 FCN을 붙여놓기만 하면 될까? 각각의 기능을 살펴보자.<br>
FCN의 첫 번째 특징은 'Pixel level Classification'이다. 이는 픽셀별로 어떤 클래스인지 구분을 하는 것을 의미하는데, 따라서 Activation Function 또한 'Per Pixel Softmax'이다. 그런데 Faster R-CNN의 모델은 이미 Classification을 통해 BBox 내부의 객체가 어떤 클래스에 해당하는지를 분류해주고 있다. 따라서 두 모델의 기능이 겹치므로 FCN은 Classification을 해 줄 필요가 없게 되고, 픽셀별로 객체가 존재하는지의 여부만 따져주면 된다. 즉 FCN의 기능은 'Per Pixel Sigmoid(Binary)'만 해주면 된다.<br>
FCN의 두 번째 특징은 'Multi Instance'이다. 마찬가지로 Faster R-CNN 모델은 이미 인스턴스 단위로 RoI를 잡아주기 때문에 이에 대해서도 걱정할 필요가 없다.<br>
즉 Mask R-CNN의 FCN 부분은 단순히 Sigmoid를 통해 픽셀별로 해당 BBox 내에서 객체가 존재하는지를 Binary로 출력해주기만 하면 되는 것이다.

### Loss Function

<img src="https://i.imgur.com/eYGZDPm.png" width="100%">

Mask R-CNN의 Loss Function은 위 그림과 같다. 기존 Faster R-CNN의 Loss Function에서 L(mask)가 추가된 형태로 이는 단순히 Binary Masking에 대한 loss값을 의미한다.

<img src="https://i.imgur.com/sGpowg6.png" width="100%">

조금 더 자세히 살펴보면 위의 슬라이드 그림을 통해 확인할 수 있다. L(mask)의 경우 모든 클래스에 대해서 loss값을 계산하지만, 결국 Classification에 의해 정해진 클래스 값의 loss값만 취하게 된다. 즉 Mask 브랜치는 클래스에 관계없이 오직 마스킹하는 법만 배운다고 말할 수 있다.

### RoI Align

Mask R-CNN의 주요 특징 중 하나는 'RoI Align'이라고 할 수 있겠다. 이전 Fast R-CNN, Faster R-CNN 모델에서 RoI Pooling의 단점을 언급했었는데, 이를 해결해주는 방법이 이것이다. 7x7 feature를 segmentation하기엔 너무 작은 사이즈이기에 feature를 pooling하는데 좀 더 정확한 방법이 필요했다.

<img src="https://i.imgur.com/aTWdcOq.png" width="100%">

위에 설명되어있는 RoI Pooling 그림과 비교해서 살펴보면 RoI Align의 장점을 더 잘 파악할 수 있다. RoI Pooling의 경우 실수 픽셀을 예측했을 때 반올림하여 Pooling을 하고, MaxPooling을 할 때에도 중앙을 정확히 자르지 못하는 경우 반올림하여 수행한다. 이런 식으로 수행하면 의도한 정보를 손실하게 되는 문제가 발생한다. 이에 대한 해결책으로 RoI Align은 실수값을 끝까지 끌어올리는 방법을 취하는데, MaxPooling을 위해 4등분을 하고, 더욱 정확하게 수행하기 위해 다시 한 번 4등분을 한다. 이 떄 이를 'Subcell'이라 부른다. 그리고 각 Subcell마다 binary interpolation을 수행하여 더욱 정확한 값을 도출해낸다. 이후 도출된 값들을 바탕으로 MaxPooling을 수행한다. 이러한 방법은 기존의 RoI Pooling 했을 떄의 정보손실을 줄여주는 역할을 함으로써 Binary Masking의 성능을 더욱 좋게 만들어주는 효과를 발휘한다.

<img src="https://i.imgur.com/dF84kK9.png" width="100%">

### RoI Issue: Zero Padding

<img src="https://i.imgur.com/kEKTAgL.png" width="100%">

추가적으로 Mask R-CNN은 RoI를 뽑을 때 Zero Padding을 해준다. 이를 통해 해당 인스턴스와 붙어있는 다른 인스턴스를 확실하게 구분시켜주어 인스턴스 예측이 용이하게 해준다.

### Network Architecture

<img src="https://i.imgur.com/BVr2bon.png" width="100%">

논문에서 제안한 Network Architecture는 위의 그림과 같이 2개의 종류로 나뉜다. 회색 음영처리부분이 각각 ResNet과 FPN을 의미하며 Mask Branch를 가지고 있다. 특이한 점은 Mask Branch로 뻗어나가는 부분의 shape이 (14, 14, 256)인 점이다.
* Faster R-CNN + ResNet
* Faster R-CNN + FPN

<img src="https://i.imgur.com/EXl2ivJ.png" width="100%">

요약하자면, Mask R-CNN은 Faster R-CNN + Binary Mask Prediction + FCN + RoIAlign이라고 할 수 있겠다. 추가적으로 Mask R-CNN의 단점은 BBox 안에서만 segmentation이 가능하기 때문에 BBox를 잘 잡아야만 성능이 좋게 나온다.

# Reference

[Deep Learning] pre-training 과 fine-tuning (파인튜닝)
* https://eehoeskrap.tistory.com/186

Object recognition을 위한 선택적 검색
* https://murra.tistory.com/25

[분석] Faster R-CNN
* https://curt-park.github.io/2017-03-17/faster-rcnn/

PR-012: Faster R-CNN : Towards Real-Time Object Detection with Region Proposal Networks
* https://www.youtube.com/watch?v=kcPAGIgBGRs&list=WL&index=8&t=0s
* https://www.slideshare.net/JinwonLee9/pr12-faster-rcnn170528

PR-057: Mask R-CNN
* https://www.youtube.com/watch?v=RtSZALC9DlU&list=WL&index=9&t=0s
* https://www.slideshare.net/TaeohKim4/pr057-mask-rcnn
