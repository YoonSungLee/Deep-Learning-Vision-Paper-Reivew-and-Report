# Crack detection using image processing: A critical review and analysis

---

https://www.sciencedirect.com/science/article/pii/S1110016817300236<br>

# Abstract
**50 research papers are taken related to crack detection, and those research papers are reviewed.**<br>
해당 논문은 리뷰 논문으로써, 기존 crack에 대한 수동 검사 방식의 한계점에 대응하여 image processing techniques를 활용한 자동화 검사 방식을 소개한다.

# 1. Introduction
**The automated crack detection can be done using some of the Non-destructive testing techniques like (i) Infrared and thermal testing, (ii) Ultrasonic testing, (iii) Laser testing, and (iv) Radiographic testing**<br>
crack을 검출하는 방법은 크게 파괴검사와 비파괴검사가 있는데, 해당 논문에서는 자동화된 crack 탐지 기술을 통해 비파괴검사의 실현을 보여준다. 특히 위의 4가지 방법이 비파괴검사의 예시라고 할 수 있다.<br>
<br>

**Some of the difficulties in the image based detection are because of the random shape and irregular size of cracks and various noises such as irregularly illuminated conditions, shading, blemishes and concrete spall in the acquired images.**<br>
image based detection의 어려움을 소개한다. 이에 대한 예시로써, crack의 불규칙적인 크기와 모양, 수집된 이미지의 불규칙한 조명, 음영, 잡티, 콘크리트 스풀 등의 다양한 노이즈 등이 존재한다. <br>
<br>

**These methods are classified into four categories, namely integrated algorithm, morphological approach, percolation-based method, and practical technique.**<br>
위의 문제를 해결하기 위한 여러 영상처리기법들이 등장하였는데, 이러한 기법들은 위의 4가지 카테고리로 분류를 할 수 있다.

# 2. Crack detection using image processing: Architecture
<img src='https://i.imgur.com/uem3bfN.png' width='100%'>

# 3. Survey
## 3.1. Camera based image processing techniques
**They have used skeletonization algorithm for the retrival of the crack segments.**<br>
crack detection은 폭을 측정하는 작업이 필수적인데, 대부분 이를 skeletonization method를 사용하여 해결한다.<br>
<br>

**The detection of the crack based upon the width and the length was completely based on the crack quantification model evaluation.**<br>
또한 crack의 폭이나 길이를 구하는 작업은 mask 위에서 수행되기 때문에, 모델의 성능에 크게 영향을 받는다. 따라서 segmentation 성능을 높이는 작업이 가장 중요하다.<br>
<br>

이후 방법론들은 camera based image processing techniques과 관련된 논문들에 대한 간단한 고찰들을 제시한다. 각 논문별로 쓰인 기법들의 목적을 파악하고, 현재 필요한 기법에 대하여 깊이 있게 연구해야 효율적인 업무를 수행할 수 있을 것이다.<br>
*my keyword*<br>
* curvature evaluation, mathematical morphology technique: noise가 많은 환경에서 crack과 비슷한 패턴들을 탐지하는 기법
* linear filtering: crack과 배경을 구분 --> threshold를 낮춘 masking에서 유용할것이라 추측
* Yang et al. have proposed an image analysis method to capture thin cracks and minimize the requirement for pen marking in reinforced concrete structural tests --> 얇은 crack 검출과 관련된 논문(6번에서 소개, paper: [Thin crack observation in a reinforced concrete bridge pier test using image processing and analysis](https://www.sciencedirect.com/science/article/abs/pii/S096599781500023X))
<br>

1.<br>
**Alam et al. have proposed a detection technique by the combination of the digital image correlation and acoustic emission.**<br>
digital image correlation이 주가 되어 표면의 정확한 측정을 수행하고, acoustic emission이 이를 보완하는 역할을 한다.<br>

2.<br>
**Iyer et al. have designed a three-step method for the crack detection from the high contrast images. The proposed method detects the crack like pattern in the noisy environment using curvature evaluation and mathematical morphology technique. It was based on mathematical morphology and curvature evaluation that detects crack-like patterns in a noisy
environment. In their study, segmentation is done defining the crack like pattern with respect to a precise geometric model. Linear filtering was performed after cross curvature evaluation to distinguish them from analogous background pattern.**<br>
해당 방법은 크게 두 가지 단계로 구분할 수 있다. 먼저 curvature evaluation과 mathematical morphology technique를 통해 noise가 많은 환경에서 crack과 비슷한 패턴들을 모두 탐지한다. 이후 linear filtering을 통해 crack과 배경을 구분하여 탐지한다.<br>

3.<br>
**Salman et al. proposed an approach to automatically distinguish cracks in digital images based on the Gabor filtering. Multidirectional crack detection can be achieved by high potential Gabor filter.**<br>
Gabor filtering은 여러 방향으로 뻗쳐진 crack을 탐지하는데에 유용한 방법론이다.<br>
<br>

4.<br>
**Sinha et al. [20] have investigated the cracks by using the two-step approach. They have developed a statistical filter design for the crack detection. After the filtering, they have got to the two-step approach at which the crack feature extraction was done locally at the first step of the pre-processing and then they have fused the images. The second step is to define the crack among the image segment by the process of cleaning and linking. They have overcome their previous work disadvantage where the morphological approach was used.**<br>
기존 morphological approach에 대한 단점을 극복한 방법으로써, 크게 두 가지 단계로 나뉜다. 먼저 자체적인 statistical filtering을 수행하고, 이후 cleaning과 linking을 통해 image segment 중에서 crack을 정의한다.<br>
<br>

5.<br>
**Talab et al. [22] have presented a new approach in image processing for detecting cracks in images of concrete structures. Here the methodology involves three steps: First; change the image to a gray image using the edge of the image and then use Sobel’s method to develop an image using Sobel’s filter for detecting cracks. Then by using suitable threshold binary image of the pixel they are categorized into the foreground and the background image. Once the images are categorized, Sobel’s filtering was used for the elimination of residual noise. After the vast filtering procedure of the image, cracks were detected using the otsu’s method. They have replaced the sober filter with the multiple median filtering in certain cases.**<br>
뭔가 굉장한 방법을 사용한 것 같은데, 최후의 방법론으로 남겨둘 만 하다. 중요한 점은 crack을 탐지하기 위한 대부분의 과정이 image processing을 통해 이루어졌다는 것이다.<br>
<br>

6.<br>
**Yang et al. have proposed an image analysis method to capture thin cracks and minimize the requirement for pen marking in reinforced concrete structural tests.**<br>
얇은 crack을 탐지하기 위해 수행했던 연구로써, 현재 풀고자 하는 문제에 가장 가까운 논문이라고 할 수 있다. 이후 소개에서는 연구에서 사용한 여러 기법들을 소개하는데, 그 종류가 굉장히 다양하기 때문에 논문을 통해 이해하는 과정이 필요하다.
