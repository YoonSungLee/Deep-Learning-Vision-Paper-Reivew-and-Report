# Segmentation Criterion

---

# 1. IoU
> IoU = inter/union = TP/(FP+TP+FN)<br>
0 <= IoU <= 1<br>
undestandable

# 2. Dice Coefficient
> dice = 2*inter/(union + inter) = 2TP/(2TP+FN+FP)<br>
0 <= dice <= 1
balance precision and recall

### 2.1. f1 score
> f1 score = 1/((1/precision)+(1/recall))
### 2.2. precision
> precision = TP/(TP+FP)
### 2.3. recall
> recall = TP/(TP+FN)

### 3. Loss
> loss = 1 - Dice<br>
0<=loss<=1