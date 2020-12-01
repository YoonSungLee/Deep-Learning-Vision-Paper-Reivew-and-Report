*해당 자료는 Stanford에서 제공하는 CS231n(2017)과 Lecture Note를 바탕으로 작성된 것임을 밝힙니다.*<br>

https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv

# Lecture 5. Convolutional Neural Networks

---



# 1. Convolutional Layer

<img src='image/cs231n018.png' width='100%'>

<img src='image/cs231n019.png' width='100%'>

parameter의 수를 계산할 때에는 bias를 명심하자.



# 2. Receptive field

한 뉴런이 한 번에 수용할 수 있는 영역

<img src='image/cs231n020.png' width='100%'>

위의 예시에서 28x28x5 filter 중의 한 영역인 5개의 뉴런은 동일한 input 공간을 보지만(receptive field가 동일하지만) 서로 다른 특징을 추출한다.



# 3. Pooling layer

<img src='image/cs231n021.png' width='100%'>

* 파라미터의 수 감소
* 공간적인 불변성(invariance)을 얻기 위함
* max pooling: 그 지역이 어디든지 어떤 신호에 대해 '얼마나' 그 필터가 활성화되었는지를 알려준다.

* pooling의 목적은 downsampling이고 코너의 값을 계산하지 못하는 경우가 없기 때문에 일반적으로 padding을 사용하지 않는다.