# Abstract

Instance Segmentation을 이해함에 있어서 그 기저가 되는 Object Detection과 Segmentation을 이해하는 것은 필수이다. 이번 Report는 Object Detection의 개념과 작동원리에 대해서 명확하게 이해하는 것이 목적이다. 본 자료는 Deeplearning.ai의 Convolutional Neural Newtorks(Course 4 of the Deep Learning Specialization) 강의 중 C4W3L01 ~ C4W3L10(총 10개 강의)를 참고했다.

# Object Localization

<img src="https://i.imgur.com/bGpdSjp.png" width="100%">

Object Detection을 이해하기 위해서는 그것의 기저인 Image Classification과 Localization을 이해해야한다. 맨 왼쪽 그림은 Image Classification으로 단순히 Image 내의 물체가 어떤 물체인지 분류해준다. 나아가 가운데 그림은 Image 내의 물체를 분류해주고 그 물체의 Bounding Box를 잡아준다. 이 때 지금까지 언급한 두 문제는 Image내의 하나의 물체만을 대상으로 한다. 만약 그 대상이 많아진다면 물체 각각에 대하여 분류해주고 Bounding Box를 잡아주어야 할 것이다. 이것이 바로 Object Detection이다.

<img src="https://i.imgur.com/79QEXs2.png" width="100%">

위에서 언급한 문제 중 'Classification with Localization' 문제를 좀 더 살펴보자. 모델은 위와 같이 구성된다. 마지막 출력층을 보면 기존 Classification 문제를 해결하기 위한 softmax 분류 layer 뿐만 아니라 물체의 Bounding Box를 출력하는 층이 포함된다. 해당 층은 총 4개의 결과를 출력하는데, Bounding Box의 중심좌표에 해당하는 bx, by, Bounding Box의 높이와 너비에 해당하는 bh, bw가 그것이다. 특히 분류를 함에 있어서 Bounding Box안에 물체가 없을 경우가 있으므로 background에 해당하는 클래스를 하나 추가해준다.

<img src="https://i.imgur.com/CTwRVx9.png" width="100%">

target label을 만드는 방법은 위와 같다. y 벡터는 물체의 존재 여부(물체가 있는지 나타내는 확률)를 표현하는 pc, Bounding Box의 정보를 표현하는 bx, by, bh, bw, 그리고 클래스 별 존재확률인 c1, c2, c3, ... 으로 구성된다. 특히 pc의 경우 물체가 있다면 1, 없다면 0의 값을 갖는데, 만약 물체가 없다면 나머지 원소들의 값이 무의믜하므로 신경쓰지 않는다. 따라서 Loss Function의 구성을 생각해보자면 다음과 같다. 먼저 물체가 존재하는 경우(y=1)는 y_hat과 y벡터의 L2 Loss를 통해 그 차이를 계산할 수 있다. 물체가 존재하지 않는 경우(y=0)는 pc의 값만 유의미하기 때문에 y_hat과 y 벡터의 pc에 해당하는 원소만 제곱 오차를 사용하여 그 차이를 계산한다. 물론 이는 가장 간편한 방법이고, 설계에 따라서 pc부분을 로지스틱 회귀 손실을 사용할 수도 있고, c1, c2, c3, ...부분을 로그 우도 손실을 사용할 수  도 있다.

# Landmark Detection

<img src="https://i.imgur.com/buozZbl.png" width="100%">

그 다음 살펴볼 내용은 Landmark Detection(특징점 검출)이다. 흔히 Face Recognition 문제나 Pose Estimation 문제가 이에 해당한다. 이는 Object Localization과는 달리 얼굴 인식에 중요한 특징점 또는 자세 추정에 중요한 특징점 등 문제를 해결하기에 중요하다고 생각하는 특징점들의 좌표를 예측하는 문제이다. 따라서 y 벡터는 물체의 존재 여부(물체가 있는지 나타내는 확률)를 표현하는 pc, 특징점들의 좌표를 표현하는 l1x, l1y, l2x, l2y, ...로 구성되어있다. 예를 들어 위의 가운데 이미지는 Face Recognition 문제에 해당하는데, 얼굴의 존재여부를 표현하는 원소, 그리고 얼굴인식에 중요하다고 생각하는 64개 특징점의 좌표를 표현하는 원소로 이루어져 총 129개의 원소를 가지는 y 벡터가 필요할 것이다. 특히 중요한 점은 특징점은 다른 이미지에 대해서도 항상 동일해야 한다는 점이다. 예를 들어 특징점 1은 항상 왼쪽 눈의 눈꼬리가 되어야 하고, 특징점 2는 항상 오른쪽 눈의 눈꼬리가 되어야 한다. 이렇듯 레이블 순서는 다른 이미지에 대해서도 항상 동일하게 구성되어 있어야 한다.

# Object Detection

<img src="https://i.imgur.com/3mVAGcC.png" width="100%">

이제 Object Detection에 대해서 살펴보자. 이는 물체의 위치를 탐지하여 BBox를 그리고 물체를 분류하는 문제라고 앞에서 언급했다. 기본적인 작동방식은 위의 그림과 같다. 먼저 전체 Image에서 자동차가 있을 만한 곳의 BBox를 모두 잡는다. 물론 자동차를 잡지 못한 BBox 또한 존재할 것이다. 이 모든 BBox를 ConvNet에 집어넣어 자동차인지를 분류하면 된다. 요약하자면, 물체가 있을만한 곳의 BBox를 모두 잡아내고, 그것들을 모두 ConvNet에 집어넣어 물체의 존재여부를 분류한다. 마찬가지로 ConvNet을 훈련시키기 위해서는 배경을 제외한 자동차만 존재하는 BBox와 그 레이블 데이터를 준비해야 할 것이다.

<img src="https://i.imgur.com/D3aQAek.png" width="100%">

그렇다면 자동차가 있을만한 곳의 BBox는 어떻게 잡는걸까? 이 방법 중의 하나로 'Sliding windows detection'이 있다. 미리 정해둔 크기의 window를 Image상에서 일정간격마다 적용시켜 수많은 BBox를 만드는 방법이다. 이후 추출한 BBox들을 ConvNet에 집어넣으면 될 것이다. 간단해보이는 방법이지만 단점이 존재한다. 이 방법은 이미지의 수많은 영역을 모두 잘라내야 하고 합성곱 신경망을 통해 이것들을 각각 계산해야 하기 때문에 시간과 비용의 문제가 발생한다. 매우 정밀한 입도 또는 이동간격을 사용한다면 모든 작은 영역을 합성곱 신경망에 통과시켜야 하기 때문에 매우 높은 계산 비용이 필요하다. 반대로 매우 큰 슬라이드 간격을 사용한다면 합성곱 신경망을 통과시켜야 하는 window의 수는 줄어들지만 물체를 제대로 탐지하지 못해 성능이 저하되는 문제가 발생한다.

# Convolutional Implementation Sliding Windows

앞서 제시한 Sliding Windows 방법은  Window 각각을 ConvNet에 입력해야 하기 때문에 시간이 오래 걸린다는 단점을 확인했다. 하지만 이는 Convolutional Implementation을 통해 그 문제를 해결할 수 있는데, 살펴보도록 하자.

<img src="https://i.imgur.com/Ubwnwb9.png" width="100%">

먼저 FC가 어떻게 Convolutional Layer로 바뀔 수 있는지 살펴보자. 위의 그림의 첫 번째는 FC Layers를 의미하고, 두 번째는 Convolutional Layers를 의미한다. FC Layers에서는 5x5x16의 feature map을 단순히 FC를 이용하여 400개의 출력을 만들어낸다. 반면에 이를 400 filter의 5x5 conv를 이용한다면 1x1x400의 출력을 만들어내어 마치 FC와 같은 기능을 하는 역할을 한다. 핵심은 1x1 conv라고 할 수 있겠다.

<img src="https://i.imgur.com/BEz4j2G.png" width="100%">

자 그럼 이 방법을 이용해서 무슨 이점을 얻을 수 있는지 살펴보자. 위의 그림의 첫 번째 내용은 14x14x3 Image에 대하여 적용한 내용으로써 이전 슬라이드와 같다. 두 번째, 세 번째의 Image에 14x14 사이즈의 Sliding Window를 적용한다고 가정해보자.<br>
두 번째 내용 기준으로, 만약 우리가  Convolutional Implementation을 사용하지 않는다면 총 4개의 Window를 ConvNet에 적용시킬것이다. 예시이기 때문에 작은 사이즈 Image에 적용했지만 만약 더 큰 사이즈의 Image라면 더 많은 수의 Window를 ConvNet에 적용시켜야 하기 때문에 시간이 오래 걸린다. 또한 각 Window는 겹치는 부분이 많아서 ConvNet이 많은 반복 계산을 수행한다는 비효율이 발생한다.<br>
Convolutional Implementation을 사용한다면 어떻게 될까? 16x16x3 Image에 첫 번째 내용과 같은 구조의 모델을 적용시킨다면 총 2x2x4의 출력을 얻을 수 있는데, 이는 곧 4개의 Window를 ConvNet에 입력시켜야하는 과정을 한 번에 수행할 수 있다는 것을 보여준다. 출력값의 (1,1)은 첫 번째 Window를 적용한 값을 나타내고, 출력값의 (1,2)는 두 번째 Window를 적용한 값을 나타낸다고 볼 수 있겠다. 즉, 연산과정에서 겹치는 반복계산을 공유하는 이점을 가질 수 있다. 그리고 각 위치에 대해서 4개의 값을 갖게 되는데, 이는 마치 Softmax함수를 수행한 것과 같은 효과를 얻을 수 있다. 즉, 각각의 Window가 어느 클래스를 나타내는지 분류하는 문제와 같다.<br>
마찬가지로 세 번째 내용을 살펴보면 Sliding Window를 통해 64개의 filter map을 ConvNet에 적용시키는 과정을 단 한 번에 수행하는 것을 보여주고 있다고 할 수 있겠다.

<img src="https://i.imgur.com/4E7ANOT.png" width="100%">

정리하자면, Convolutional Implementation을 이용하여 각각의 Window를 ConvNet에 집어넣는 과정을 단 한번에 수행할 수 있는 모델을 만들 수 있다. 하지만 이 알고리즘은 여전히 문제를 가지고 있는데, BBox의 위치가 정확하지 않을 수 있다는 것이 그것이다.

# Intersection Over Union

<img src="https://i.imgur.com/5r1L3Sr.png" width="100%">

이번엔 Object Detection이 잘 작동하는지 확인하는 방법, 즉 성능평가방법에 대해서 알아보자. 그 명칭을 먼저 말하자면 'IoU(Intersection over Union)'이라고 한다. 명칭이 꽤 직관적으로 알 수 있도록 지었다는 느낌이 든다. 위의 그림을 예로 들면, 물체의 실제 BBox는 빨간색 박스이고 예측 BBox는 보라색 박스이다. Union은 두 박스의 모든 영역으로써 초록색으로 빗금친 영역을 의미하고, Intersection은 두 박스의 공통적인 영역으로써 노란색으로 빗금친 영역을 의미한다. 이 두 영역을 나눠주기만 하면 IoU를 구할 수 있다. 만약 예측 BBox가 실제 BBox를 정확하게 예측했다면 이 값은 1이 될 것이고, 공통부분 없이 완전히 틀리게 예측했다면 이 값은 0이 될 것이다. 관례적으로 컴퓨터 비전분야에서는 이 값이 0.5 이상만 되면 성능이 괜찮다라고 평가하여 임계값을 0.5로 설정하고는 한다.

# Non-max Suppression

<img src="https://i.imgur.com/kC7USCe.png" width="100%">

지금까지 배운 Object Detection 방식은 단점이 하나 있다. 초기에 지정해준 수많은 BBox가 학습을 거쳐 물체를 가리킬텐데, 중복된 BBox들이 많을것이라는 점이다. 예를 들어 위의 왼쪽 그림처럼 19x19개의 그리드를 ConvNet에 집어넣는다고 해보자. 자동차 주변의 몇몇 그리드는 예측한 BBox가 자동차라고 예측하고 검출할것이다. 이러한 몇몇 그리드는 결국 같은 자동차를 검출하는 셈이 된다. 이런 중복된 BBox에 대해서 제거를 해 주는 것이 명확한 결과를 준다. 이를 해결하는 방법이 Non-max Supression이라고 한다.

<img src="https://i.imgur.com/M4H8vVe.png" width="100%">

위의 그림으로 예를 들어보자. 왼쪽 자동차를 0.8로 예측한 BBox와 0.7로 예측한 BBox는 결국 중복이므로 하나를 제거해야하고, 오른쪽 자동차를 각각 0.6, 0.7, 0.9로 예측한 BBox들도 중복이므로 하나만 남기고 제거해야한다. 이를 제거할때는 BBox 내에 물체가 있다고 예측하는 확률인 Pc가 가장 높은 것을 남기고 제거하는것이 상식적이다. 즉, 이 경우에 NMS를 적용하고 나면 0.8로 예측한 BBox와 0.9로 예측한 BBox가 남겨질것이라고 예상할 수 있다.

<img src="https://i.imgur.com/JOHkgtc.png" width="100%">

그럼 Non-max Supression 작동방식에 대해 살펴보자. 위 그림에서는 하나의 물체를 검출하는 상황이라고 가정하자. 먼저 모든 BBox에 대하여 Pc가 0.6 이하인 것들은 다 버린다. 이는 물체를 검출했다고 보기 어렵기 때문이다. 이후 남아있는 박스들에 대해서 NMS를 수행한다. 특정 클래스(위 그림에서는 하나의 클래스만 존재)를 기준으로 가장 높은 Pc값을 가지고 있는 BBox를 선택한다. 이후 해당 BBox와 남아있는 BBox를 비교하여 IoU가 0.5 이상인 것들은 중복이라고 판단하고 버린다. 마찬가지로 남이있는 박스들에 대해서 또 다시 가장 높은 Pc값을 가지고 있는 BBox를 선택하는 과정을 반복한다. 만약 보행자, 자동차, 오토바이 같은 세 개의 물체를 검출하려 한다면 결과 벡터는 추가적인 세 개의 요소를 가질 것이고 각각의 결과 클래스에 대해 독립적으로 세 번의 NMS를 해야 한다.

그런데 이 방식에는 문제점이 하나 있다고 생각한다. 문제점과 그에 대한 해결방안을 제시하기에 앞서 이 방법이 최적인지는 아직 더 살펴봐야한다는 것을 미리 언급하겠다. 몇몇 경로를 통해서 조사한 바에만 국한되어있음을 밝힌다. 문제점은 뭉쳐져 있는 물체에 대한 검출을 할 경우에 발생한다. 이해를 돕기 위해 사람을 검출하는 경우를 예로 들겠다. 사람들이 서로 어깨동무를 하거나 포옹을 하는 경우 등 사람들이 뭉쳐있는 경우는 얼마든지 발생한다. 만약 이러한 상황에서 NMS로 검출을 한다면 가장 높은 Pc값을 갖는 BBox를 검출한 뒤에 실제로는 옆 사람을 검출한 BBox를 IoU 임계값때문에 제거하는 상황이 발생한다. 즉, 뭉쳐져 있는 객체에 대한 검출이 불가능할것이다. 이에 대한 몇 가지 조사를 한 결과로써는 임계값 조정이 그 해결책이 될 수 있겠다. 어떤 문제를 해결하고자하는지에 따라 IoU 임계값을 조절할 수 있다. 임계값이 높다면 뭉쳐져 있는 물체를 검출을 잘 할 수 있지만 오히려 하나의 물체에 2개의 BBox를 잡을 수도 있다. 즉, BBox의 개수가 상대적으로 많아진다. 반대로 임계값이 낮다면 하나의 물체에 확실한 하나의 BBox를 잡을 수 있지만 조금만 옆에 있던 물체의 BBox 또한 잡지 못할수도 있다. 즉, BBox의 개수가 상대적으로 적어진다. 결국 IoU의 임계값은 하이퍼파라미터로써 설계자가 조정해야하는 값이다. 만약 더 좋은 방법이 있다면 기회가 될 때 업로드하도록 하겠다.

NMS와 이에 대한 문제점을 해결하기 위한 방안에 대해서 이해하는데에 시간투자를 한 만큼 코드에 대한 이해까지 하기 위해 아래에 NMS를 구현한 코드를 몇 가지 소개하겠다.

<img src="https://i.imgur.com/ohFswWp.png" width="100%">

```python
def nms(boxes, probs, threshold):
    """Non-Maximum supression.
    Args:
      boxes: array of [cx, cy, w, h] (center format)
      probs: array of probabilities
      threshold: two boxes are considered overlapping if their IOU is largher than this threshold
      form: 'center' or 'diagonal'
    Returns:
      keep: array of True or False.
    """
    
    order = probs.argsort()[::-1]    # np.argsort(): 작은값부터 순서대로 데이터의 index를 반환
    keep = [True]*len(order)

    for i in range(len(order)-1):    # 마지막 하나는 수행 X
        ovps = batch_iou(boxes[order[i+1:]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j+i+1]] = False
    return keep


def batch_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another box.
    
    Args:
        box1: 2D array of [cx, cy, width, height].
        box2: a single array of [cx, cy, width, height]
    Returns:
        ious: array of a float number in range [0, 1].
  """
    
    # np.maximum: 두 개의 배열을 취하여 요소별 최대 값을 계산(아래 셀 예시 참고)
    lr = np.maximum(
        np.minimum(boxes[:,0]+0.5*boxes[:,2], box[0]+0.5*box[2]) - \
        np.maximum(boxes[:,0]-0.5*boxes[:,2], box[0]-0.5*box[2]),
        0
    )
    tb = np.maximum(
        np.minimum(boxes[:,1]+0.5*boxes[:,3], box[1]+0.5*box[3]) - \
        np.maximum(boxes[:,1]-0.5*boxes[:,3], box[1]-0.5*box[3]),
        0
    )
    inter = lr*tb
    union = boxes[:,2]*boxes[:,3] + box[2]*box[3] - inter
    return inter/union
```
<br>
<br>
Reference) https://github.com/BichenWuUCB/squeezeDet/blob/master/src/utils/util.py

```python
import numpy as np

a = np.array([3, 6, 1])
b = np.array([4, 2, 9])

print(np.maximum(a,b))
```

```python
c = np.array([3, 6, 1])
d = np.array([4])

print(np.maximum(c,d))
```

<img src="https://i.imgur.com/BuGrKOu.png" width="100%">

```python
def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou
```
<br>
<br>
Reference) http://ronny.rest/tutorials/module/localization_001/iou/

# Anchor Boxes

지금까지 배운 Object Detection 방식은 또 다른 단점을 가지고 있다. 각각의 격자 셀이 오직 하나의 물체만 감지할 수 있다는 점이다. 하지만 실제 사진들은 아래 그림처럼 여러 물체가 겹쳐서 찍힐 수도 있다. 특히 겹쳐진 물체들의 중심점이 비슷한 격자 셀에 위치한다면 해당 격자 셀은 하나의 물체에 대한 BBox만을 감지하므로 다른 물체에 대한 BBox의 성능이 그리 좋지 않을것이다. 이를 해결하기 위한 방법이 'Anchor Box'로써, 두 물체가 한 격자 셀에 나타날 경우를 다루는 방법이다.

<img src="https://i.imgur.com/FUFd3RT.png" width="100%">

지금까지 우리는 y 벡터를 왼쪽과 같이 정의했다. 만약 각 격자 셀마다 두 개의 물체를 감지하고 싶다고 해보자. 그렇다면 y 벡터를 수정하여 두 개의 BBox 즉, anchor Box의 정보를 모두 담도록 해야한다. 이를 고려한 y 벡터를 오른쪽과 같이 정의할 수 있다. 방법은 간단한데, 8개의 정보를 하나의 anchor box로 생각하여 그와 동일한 8개의 정보를 늘려주면된다. 만약 각 anchor box의 초기값을 위의 그림과 같이 하나는 길게, 하나는 넓게 설정한다면 각 물체를 더 잘 검출해낼것이다.

<img src="https://i.imgur.com/VWKVbjZ.png" width="100%">

anchor box를 사용하기 전에는 훈련 세트 사진에 있는 각 물체들은 물체의 중심점이 있는 격자 셀에 배정되었다. anchor box를 사용하게 된다면 각 물체는 이전과 같이 중심점이 있는 셀에 배정되는 것은 같지만, 이제는 물체의 모양과 가장 높은 IoU를 가지는 격자 셀+anchor box에 배정된다. 특히 각 격자 셀마다 가지고 있는 정보가 8차원인 이유는 세 개의 물체 클래스가 있기 때문이다. 만약 더 많은 물체가 있다면 y의 차원도 더 커진다.

<img src="https://i.imgur.com/ZpeBiAJ.png" width="100%">

anchor box를 사용한 예를 한 번 살펴보자. anchor box1의 초기값을 길게, anchor box2의 초기값을 넓게 설정했다. 이를 학습하여 해당 이미지를 모델에 돌린 결과는 가운데 y 벡터와 같다. anchor box1은 보행자를 감지하여 c1이 1과 가까운 값을 가지고, anchor box2는 자동차를 감지하여 c2가 1과 가까운 값을 가진다. 만약 이 경우에 자동차만 있는 경우라면 anchor box2에 해당하는 부분은 동일하지만, anchor box1에 해당하는 부분은 아무것도 감지하지 못해 Pc가 0의 값을 갖고 나머지 값들은 의미가 없어진다.

# YOLO Algorithm

지금까지 다룬 내용들이 Object Detection의 중요하고도 기본적인 개념이라 할 수 있겠다. 이러한 개념들을 바탕으로 논문에 기재된 모델이 그 유명한 YOLO 이다. 지금까지의 내용을 짚어보면서 YOLO Algorithm에 대해 살펴보겠다.

<img src="https://i.imgur.com/kTylUB3.png" width="100%">

먼저 Training data를 어떻게 구성하는지 살펴보겠다. 만약 위의 그림처럼 우리는 보행자, 자동차, 오토바이를 검출하는 모델이 필요하고 이를 바탕으로 3x3 격자 셀과 2개의 anchor box를 구성하는 모델을 만든다고 해보자. 그렇다면 y 벡터는 3x3의 격자 셀, 2개의 anchor box, 8개의 정보(Pc, bx, by, bh, bw, c1, c2, c3)를 포함해서 총 3x3x2x8로 구성된다고 할 수 있겠다. 특히 위 그림에서 표시된 y 벡터는 각 격자 셀마다 가지고 있는 정보를 의미한다. 예를 들어 첫 번째 격자 셀은 아무 물체도 없기 때문에 각 anchor box의 Pc는 0값을 가지고 나머지 값들은 의미가 없다. 반면에 자동차가 포함된 격자 셀은 두 개의 anchor box 중에 자동차와 유사한 anchor box의 값은 Pc는 1, bx, by, bh, bw는 자동차 BBox의 위치, c2는 1값을 갖는다. 즉, 예를 들어 100x100x3의 이미지를 ConvNet에 집어넣는다고 하면 3x3x2x8로 이루어진 결과를 얻을 수 있다. 이를 공부하면서 든 의문은 training datset의 구성인데, 실제로 이렇게 많은 정보를 가진 training dataset을 어떻게 만드는지이다. 격자 셀이 많다면 그만큼 y 벡터는 많아질 것이고, 격자 셀마다 배경만 있는 셀이 대부분일텐데 training dataset을 만드는 것이 쉽지 않을 것이라 생각한다. 실제 training dataset의 구성을 살펴보면서 느낌을 얻을 필요가 있고, 이 y 벡터를 잘 구성하게 해주는 프로그램이 있는 것으로 알고있는데 그것을 활용해 볼 필요가 있다. 그 과정들을 통해 모델이 작동하는 방식을 좀 더 구체적으로 상상할 수 있을 것이라 생각한다.

<img src="https://i.imgur.com/WjBPyIP.png" width="100%">

training dataset을 잘 이해했다면 prediction이 어떻게 진행되는지는 마찬가지로 이해하기 쉬울 것이다. 이에 대한 설명은 생략한다.

<img src="https://i.imgur.com/AHYUYaj.png" width="100%">

마지막으로 모델이 예측한 BBox들에 대해 NMS를 수행하는데, 위의 그림을 통해 그 방법을 자세히 살펴보도록 하겠다. 만약 두 개의 anchor box를 사용한다면 9개의 격자 셀 각각에 대해서 두 개의 BBox가 도출될 것이다. 그 중 일부는 아주 낮은 확률 Pc를 가지겠지만 어쨌든 9개의 격자 셀 모두 두 개의 BBox를 갖는다. 어떤 BBox는 격자 셀의 밖으로 나와 있을 수도 있다. 다음으로 낮은 확률 예측을 제거한다. 신경망이 물체가 거기 있을지도 모른다고 했더라도 그것들을 제거한다. 마지막으로 만약 우리가 보행자, 자동차, 오토바이의 세 가지 클래스를 감지하려고 한다면 각각의 세 클래스에 대해서 그 클래스로 감지된 물체에 대해서 독립적으로 NMS를 실행한다. 보행자 클래스의 예측에 대해 NMS를 실행하고 자동차 클래스와 오토바이 클래스에 대해서도 NMS를 실행한다. 최종 예측을 얻기 위해서 총 세 번 실행한다. 이것의 결과값은 아마도 모든 자동차와 보행자가 검출된 것이다. 이것이 YOLO 알고리즘이다.

# Region Proposals

YOLO 알고리즘과는 다른 방법으로 Object Detection을 이용한 모델들도 있다. 그 방법을 'Region Proposals'이라고 한다. 이는 ConvNet을 실행할 몇 개의 지역만을 고르는 방법이다. 슬라이딩 윈도를 모든 윈도에 적용하지 않고 몇 개의 윈도만 골라서 거기에만 ConvNet을 실행하는 방식이라 볼 수 있겠다. 이에 해당하는 모델이 R-CNN 계열의 모델인데, 이는 전에 다룬 바 있는 모델들이기 때문에 설명은 생략하기로 하겠다. 이 강의에 대한 슬라이드와 R-CNN 관련 Report는 아래 링크에 걸어두도록 하겠다.

<img src="https://i.imgur.com/0pTXxge.png" width="100%">

<img src="https://i.imgur.com/2EdA4Z9.png" width="100%">

R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN Report
* https://github.com/YoonSungLee/Mask-R-CNN-Project_AI-Innovation-Square
* https://github.com/YoonSungLee/Mask-R-CNN-Project_AI-Innovation-Square/blob/master/Mask%20R-CNN%20Report.ipynb

# Reference

deeplearning.ai
* https://www.deeplearning.ai/
* https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w

squeezeDet
* https://github.com/BichenWuUCB/squeezeDet/blob/master/src/utils/util.py

Intersect over Union (IoU)
* http://ronny.rest/tutorials/module/localization_001/iou/

다양한 IOU(Intersection over Union) 구하는 법
* https://gaussian37.github.io/math-algorithm-iou/
