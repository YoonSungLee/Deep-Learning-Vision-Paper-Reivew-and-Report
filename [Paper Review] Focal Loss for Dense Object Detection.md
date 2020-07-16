# Focal Loss for Dense Object Detection

paper(17.08) https://arxiv.org/abs/1708.02002
code None

---

# Abstract

**"We discover that the extreme foreground-background class imbalance encountered during training of dense detectors is the central cause."**
Object Detection Model은 크게 1-stage 계열과 2-stage 계열로 구분할 수 있다. 흔히 2-stage 계열의 모델이 1-stage 계열의 모델보다 speed는 낮지만 accuracy가 높은 특징을 가지고 있다. 저자는 1-stage 계열의 모델의 accuracy가 상대적으로 낮은 이유를 class imbalance 문제 때문이라고 발견했다.

**"We propose to address this class imbalance by reshaping the standard cross entropy loss such that it down-weights the loss assigned to well-classified examples. Our novel Focal Loss focuses training on a spares set of hard examples and prevents the vast number of easy negatives from overwhelming the detector during training."**
따라서 기존에 존재하던 cross entropy loss function을 수정하여 학습을 잘 하지 못하는 dataset에 가중치를 부여하도록 하는 loss function이 이 논문의 주제인 Focal Loss이다.

