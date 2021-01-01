# GAN survey



---



# 1. GAN: Generative Adversarial Networks (꼼꼼한 딥러닝 논문 리뷰와 코드 실습)

Reference) [[link]](https://www.youtube.com/watch?v=AVvlDmhHgC4)

## 1. 이미지 데이터에 대한 확률 분포

* 이미지 데이터는 다차원 특징 공간의 한 점으로 표현된다.
  * 이미지의 분포를 근사하는 모델을 학습할 수 있다.
  * 예를 들어 사람의 얼굴에는 통계적인 평균치가 존재하는데, 모델은 이를 수치적으로 표현할 수 있게 된다.

* 즉, 이미지 데이터에 대한 확률 분포는 이미지에서의 다양한 특징들이 각각의 확률 변수가 되는 분포(다변수 확률 분포, multivariate probability distribution)를 의미한다.



## 2. 생성 모델(Generative Models)

* 실존하지 않지만 있을 법한 이미지(+ 자연어, 오디오 등 모든 데이터 포함)를 생성할 수 있는 모델

* A statistical model of the joint probability distribution
* An architecture to generate new data instances



### 생성 모델의 목표

* 이미지 데이터의 분포를 근사하는 모델 G를 만드는 것
* 모델 G가 잘 동작한다는 의미는 원래 이미지들의 분포를 잘 모델링할 수 있다는 것을 의미
* 모델 G는 원래 데이터(이미지)의 분포를 근사할 수 있도록 학습



## 3. GAN

<img src='Image/GAN001.PNG' width='100%'>