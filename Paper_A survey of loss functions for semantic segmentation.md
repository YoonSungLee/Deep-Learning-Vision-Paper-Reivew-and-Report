# A survey of loss functions for semantic segmentation
IEEE, Shruti Jadon, 2020.09.03<br>
Paper) https://arxiv.org/abs/2006.14822<br>
GitHub) https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions<br>

---

# Abstract
**In this paper, we have summarized some of the well-known loss functions widely used for Image Segmentation and listed out the cases where their usage can help in fast and better convergence of a model. Furthermore, we have also introduced a new log-cosh dice loss function and compared its performance on NBFS skull-segmentation open source data-set with widely used loss functions. We also showcased that certain loss functions perform well across all data-sets and can be taken as a good baseline choice in unknown data distribution scenarios.**<br>
loss function의 종류는 굉장히 다양하며, biased data나 sparse segmentation 등 각기 특정 상황에 놓인 task를 수행하는데에 유리한 특징들을 가지고 있다. 해당 논문은 Image Segmentation 분야에서 잘 알려진 여러 loss function들을 survey한다. 또한 저자의 새로운 log-cosh dice loss function을 소개하면서 다른 loss function과 그 성능을 비교한다. 이러한 과정 속에서 특정 loss function들은 dataset의 분포나 형태에 구애 받지 않고 general한 성능을 보이는 것들도 존재한다고 소개한다. 만약 이러한 loss function의 존재가 증명된다면, segmentation 업무를 수행함에 있어서, 제일 먼저 실험해볼 수 있는 initial function으로 사용할 수 있을 것이다.<br>
<br>

# 1. Introduction
<img src='https://i.imgur.com/L4YhueE.png' width='100%'>
<br>

# 2. Loss Function
**In this paper, we have focused on Semantic Segmentation instead of Instance Segmentation, therefore the number of classes at pixel level is restricted to 2.**<br>
해당 논문은 segmentation 분야 중에서도 pixel level이 foreground와 background 2로 제한되는 semantic segmentation에 초점이 맞추어져 있다.

## A. Binary Cross-Entropy
<img src='https://i.imgur.com/hakTeQ9.png' width='100%'>
<img src='https://i.imgur.com/cT6KujK.png' width='100%'>

## B. Weighted Binary Cross-Entropy
<img src='https://i.imgur.com/yrd0vAb.png' width='100%'>
**It is widely used in case of skewed data as show in figure 1.**<br>

## C. Balanced Cross-Entropy
<img src='https://i.imgur.com/U9F5rb2.png' width='100%'>
β = number of negative samples/ total number of samples. In other words, β is the fraction of the sample which is dominant in a dataset. 1 — β denoting the fraction of the other class(obvious!).<br>
*Reference) https://towardsdatascience.com/neural-networks-intuitions-1-balanced-cross-entropy-331995cd5033*

## D. Focal Loss
<img src='https://i.imgur.com/oYkj4Ko.png' width='100%'>
**It works well for highly imbalanced class senarios, as shown in fig 1.**

## E. Dice Loss
<img src='https://i.imgur.com/6uYFGDR.png' width='100%'>

## F. Tversky Loss
<img src='https://i.imgur.com/lNiKYqR.png' width='100%'>
**Tversky idex(TI) can also be seen as an generalization of Dices coefficient. It adds a weight to FP(false positives) and FN(false negatives) with the help of beta coefficient.**

## G. Focal Tversky Loss
<img src='https://i.imgur.com/k7vA1ez.png' width='100%'>
**Similar to Focal Loss, which focuses on hard example by down-weighting easy/common ones. Focal Tversky loss also attempts to learn hard-examples such as with small ROIs(region of interest) with the help of gamma coefficient as shown below.**

## H. Sensitivity Specificity Loss
<img src='https://i.imgur.com/gyoakpo.png' width='100%'>

## I. Shape-aware Loss
<img src='https://i.imgur.com/3zj4o25.png' width='100%'>
**Generally, all loss functions work at pixel level, however, Shape-aware loss calculates the average point to curve Euclidean distance among points around curve of predicted segmentation to the ground truth and use it as coefficient to cross-entropy loss function.**

## J. Combo Loss
<img src='https://i.imgur.com/S6G9vku.png' width='100%'>
**Combo loss is defined as a weighted sum of Dice loss and a modified cross entropy.**

## K. Exponential Logarithmic Loss
<img src='https://i.imgur.com/EbSdnFw.png' width='100%'>
**Wong et al. proposes to make exponential and logarithmic transforms to both Dice loss an cross entropy loss so as to incorporate benefits of finer decision boundaries and accurate data distribution.**

## L. Distance map derived loss penalty term
(...skip...)

## M. Hausdorff Distance Loss
(...skip...)

## N. Correlation Maximized Structural Similarity Loss
(...skip...)

## O. Log-Cosh Dice Loss
**Log-Cosh approach has been widely used in regression based problem for smoothing the curve.**
<img src='https://i.imgur.com/46AXPBM.png' width='100%'>

## Summary
<img src='https://i.imgur.com/AIYQjXu.png' width='100%'>

# 3. Experiments
<img src='https://i.imgur.com/1QzyP02.png' width='100%'>
해당 논문에서는 Evaluation Metrics로 **Dice Coefficient**, **Sensitivity**, **Specificity**를 모두 사용하였다. 일반적으로 모델의 성능 평가지표로써 Dice Coefficient만 사용하는 것으로 알고 있었으나, task에 맞게 선택하는 자세가 필요하다. 만약 이러한 판단이 어렵다면, 위에 제시된 3가지에 대하여 모두 비교하는 방식으로 진행할 수 있다.

# 4. Conclusion
(...skip...)