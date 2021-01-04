# Generative Adversarial Nets

Paper) https://arxiv.org/abs/1406.2661<br>
Reference) GAN: Generative Adversarial Networks (꼼꼼한 딥러닝 논문 리뷰와 코드 실습) [[link]](https://www.youtube.com/watch?v=AVvlDmhHgC4)

---

# Abstract

*a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.*<br>
GAN은 data의 분포를 학습하는 G(생성자)와 data가 real인지(실제 데이터인지) fake인지(G로부터 생성되었는지)를 추정하는 D(판별자)로 이루어진 개념이다.



# 1. Introduction

*In the proposed adversarial nets framework, the generative model is pitted against an adversary: a discriminative model that learns to determine whether a sample is from the model distribution or the data distribution.*<br>
GAN은 G와 D가 적대적으로 학습한다. D는 특정 샘플이 data distribution에서 추출된 것인지(real), model distribution에서 추출된 것인지(fake) 잘 구분하는 방향으로 학습한다. 반면에 G는 model distribution으로 데이터를 생성하여 D가 real이라고 인식하게끔 하는 방향으로 학습한다. 즉, G의 model distribution은 data distribution과 유사한 방향으로 학습이 이루어진다.<br>
<br>

또한 저자는 GAN의 방법이 기존의 approximate inference나 Markov chain 기법을 적용하지 않고, 오직 multilayer preceptron으로 모델을 구성했다는 점을 강조한다.<br>



# 2. Realted work

GAN 모델이 발표되기 전에 해당 분야에 대한 연구 현황을 나타내고 있다. 이 분야에 대한 background가 부족하여 skip한다.



# 3. Adversarial nets

<img src='Image/GAN003.PNG' width='100%'>

위 수식에 대한 설명은 Report에 이미 다룬 바 있으므로, 그 내용을 그대로 차용함을 밝힌다.

* V(D, G)에 대하여 G는 이 값을 minimize하려고 하고, D는 maximize하려고 한다. 이러한 목표를 봤을 때 G와 D 각각은 V(D, G)를 어떤 방향으로 이끌어가는지 생각하는 것이 중요하다.
* D 관점에서 봤을 때, logD(x)를 maximize하려고 하므로 D(x)는 1에 가까운 값을 얻으려고 할 것이다. 또한 log(1-D(G(z)))를 maximize하려고 하므로 1-D(G(z))는 1에 가까운 값 즉, D(G(z))는 0에 가까운 값을 얻으려고 할 것이다. 이를 해석하자면 D는 x를 실제(Real)라고 잘 분류하고, G(z)(G가 z를 통해 만든 가짜 데이터)를 가짜(Fake)라고 잘 분류하도록 학습이 된다.
* G 관점에서 봤을 때, log(1-D(G(z)))를 minimize하려고 하므로 1-D(G(z))는 0에 가까운 값 즉, D(G(z))는 1에 가까운 값을 얻으려고 할 것이다. 이를 해석하자면 G는 z를 통해 생성한 데이터를 D가 실제(Real)라고 잘 분류하도록 학습이 된다. 즉, 가짜 데이터셋을 잘 만들도록 학습이 된다.

<br>

<img src='Image/GAN004.PNG' width='100%'>

논문의 저자는 GAN의 학습 과정을 쉽게 이해하도록 하기 위해 위와 같은 그림을 제시한다. z는 uniform distribution이나 gaussian distribution과 같은 임의의 분포이고, x는 기존 dataset(domain)의 영역이다. z에서 x로 매핑하는 과정을 G가 담당한다. 검정색 점의 분포는 기존 dataset의 distribution을 의미하고, 초록색 선의 분포는 G가 만들어낸 dataset의 distribution을 의미한다. 이 때 기존 dataset은 무한이 아니기 때문에 '점'으로 표현하고, 반면에 G가 만들어내는 dataset은 연속적으로 만들어낼 수 있기 때문에 '선'으로 표현한다. 그리고 파란색 선은 D가 특정 dataset point를 기존 dataset의 distribution에서 추출한 것이라고 판별하는 estimation을 의미한다.<br>
학습을 시작하기 전에는 (a)와 같은 형태를 띤다. G가 학습되지 않은 단계이기 때문에, D는 어느 정도 예측을 하는 수준임을 확인할 수 있다. (먼저 D를 학습하는 과정을 거치는데, 이를 통해 D는 특정 dataset point가 어떤 distribution에서 도출된 것인지(real or fake) 잘 판별하게 되어 (b)와 같은 형태로 바뀐다. 이후 G를 학습하면, G의 model distribution이 점점 data distribution과 유사하지기 때문에 (c)와 같은 형태를 띤다. D와 G의 학습 과정을 지속적으로 반복하면, G는 더더욱 실제 data와 유사한 data를 생성해내기 때문에 결국 D의 추정은 항상 1/2를 도출한다. (d)가 이에 해당한다.<br>
<br>



*Optimizing D to completion in the inner loop of training is computationally prohibitive, and on finite dataset would result in overfitting. Instead, we alternate between k steps of optimizing D and one step of optimizing G. This results in D being maintained near its optimal solution, so long as G changes slowly enough.*<br>
Question) 이해 필요<br>
<br>



*Rather than training G to minimize log(1 - D(G(z))) we can train G to maximize logD(G(z)). This objective function results in the same fixed point of the dynamics of G and D but provides much stronger gradients early in learning*<br>
Question) 이해 필요<br>
<br>



# 4. Theoretical Results

To Do~