# 3차원 의료 영상 분할 평가 지표에 관한 고찰(Review of Evaluation Metrics for 3D Medical Image Segmentation)
김장우, 김종효<br>
2017<br>
*Reference) https://kmbase.medric.or.kr/Main.aspx?d=KMBASE&i=1147120170230010014&m=VIEW*

---

# 서론

**자동 영상 분할 방법에는 전통적인 특성 기반(feature-based) 방법과 텍스처 기반(texture-based) 방법이 있다. (...skip...) 이 외에도 모델 기반 방법과 Atlas 기반 방법이 있으며, 최근에는 신경망(neural network) 기반 방법이 새로이 고안되었다.**<br>
<br>

**분할 오차를 정량적으로 나타내기 위해 다양한 평가지표를 사용하며 평가지표의 필수조건을 다음과 같이 3가지로 나눌 수 있다. 두 비교 결과의 유사도를 나타내는 정확도(accuracy), 항상 일정한 결과를 도출해내는 반복성(repeatabilityt), 계산 시간이 길지 않는 효율성(efficiency).**<br>
<br>

# 재료 및 방법

## Overlap 기반 평가
~~~
민감도(sensitivity)
특이도(specificity)
위양성률(false positive rate)
위음성률(false negative rate)
F-Measure(FMS)
Dice similarity coefficient(DICE)
Jaccard index(JAC)
global consistency error(GCE)
~~~

1.민감도(sensitivity)와 특이도(specificitty)
<img src='https://i.imgur.com/pLIjWNK.png' width='100%'>
* gold standard의 관심영역을 자동 분할 결과가 관심영역으로 할당하고 배경 영역은 배경으로 할당했을 확률을 의미하는 직관적인 평가 지표
* 분할 영역이 작은 경우에 오차를 더 크게 반영하여 분할 영역 크기에 민감하다는 단점
<br>

2.위양성률(false positive rate)과 위음성률(false negative rate)
<img src='https://i.imgur.com/oNLgCUb.png' width='100%'>
<br>

3.F-Measure(FMS)
<img src='https://i.imgur.com/36jyzqq.png' width='100%'>
* 정보의 복원정도를 나타내는 지표로 정확도(precision)와 민감도의 상호 작용
* beta의 값이 1일 때를 주로 사용하며 이는 Dice similarity coefficient와 같다.
<br>

4.Dice similarity coefficient
<img src='https://i.imgur.com/FcjobhU.png' width='100%'>
* 영상 분할 평가에 쓰이는 가장 대표적인 지표
<br>

5.Jaccard index(JAC)
<img src='https://i.imgur.com/jIhE3Lh.png' width='100%'>
* 두 분할 결과의 합을 교차 값으로 나눈 것
* Dice와 상응 --> Dice와 JAC를 동시에 평가지표로 사용하는 것은 의미가 없다.
<br>

6.Global consistency error (GCE)
* 두 영상 분할 간의 오차를 측정하는 지표
* 오차를 측정해야만 하는 경우가 아니라면 자주 사용되지는 않는다.

## NCC 부피 기반 평가 지표
~~~
체적 유사도(volumetric similarity)
~~~

1.체적 유사도(volumetric similarity)
<img src='https://i.imgur.com/LnhPCm6.png' width='100%'>
* 두 분할 결과각각의 분할 객체 부피를 비교하는 방식
* 오직 분할 결과의 부피만을 고려하므로 반드시 비교 대상 결과 간의 정렬이 선행되어야만 한다.

## 정보 이론 기반 평가 지표
~~~
상호 정보량(mutual information)
정보 변화량(variation of information)
~~~
(...skip...)

## 확률 기반 평가 지표
~~~
계층 간 상관도(interclass correlation, ICC)
확률 거리(Probabilistic Distance, PBD)
Cohens kappa
AUC (Area under ROC curve)
~~~
(...skip...)

## 공간상 거리 기반 평가 지표
~~~
Hauuse-dorff 거리
평균 Hausdorff 거리
Mahalanobis 거리
~~~

## Paricounting 기반 평가 지표
~~~
Rank index
Adjusted Rank index
~~~
(...skip...)

<img src='https://i.imgur.com/ySbmaTE.png' width='100%'>
# 결과
1.단일 평가 지표만으로는 영상 분할 결과의 단편적인 부분을 전체로 확대 해석하거나 평가 지표에 반영되지 않는 부분은 놓칠 수 있다. 따라서 다양한 평가 지표를 통해 결론을 도출해야 영상 분할 결과를
바르게 해석할 수 있다.
2.Case별 평가 지표 선택 가이드
* 자동 분할 결과의 경계 정확도(accuracy)가 가장 중요한 경우 --> 공간 기반 평가 지표(평균 Hausdorff 거리) 적합, 체적 유사도 평가 지표는 지양
* 분할 구간이 주위 배경에 비해 너무 작은 경우(전체 크기 대비 5% 이하의 분할 구간만이 존재하는 경우) --> overlap 기반 평가 지표보다는 공간 기반 평가 지표로 분할 결과를 평가
* 분할 경계가 복잡한 경우 --> Hausdorff 거리와 평균 Hausdorff 거리가 적합
* FP 혹은 FN을 포함하더라도 모든 true 구간을 놓쳐서는 안 되는 경우 --> 상호 정보량, 위양성률
* 데이터의 이상치가 존재하는 경우 --> 이상치에 민감한 Hausdorff 거리는 지양

# 고찰 및 결론
(...skip...)