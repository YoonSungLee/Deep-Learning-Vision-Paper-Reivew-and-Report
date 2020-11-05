*해당 자료는 Stanford에서 제공하는 CS231n(2017)과 Lecture Note를 바탕으로 작성된 것임을 밝힙니다.*<br>

https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv

# Lecture 3. Loss Functions and Optimization

---

# 1. 손실 함수(loss function)

w를 입력으로 받아서 각 스코어를 확인하고 이 W가 지금 얼마나 거지같은지를 정량적으로 말해주는 것

### 1. Multiclass SVM loss

<img src='Image/cs231n008.png' width='50%'>

<img src='Image/cs231n009.png' width='100%'>



Q) At initialization W is small so all s = 0. What is the loss?<br>

A) (클래스의 수) - 1. Loss를 계산할 때 정답이 아닌 클래스를 순회한다. 그러면 C-1 클래스를 순회하게 된다. 비교하는 두 스코어가 거의 비슷하니 Margin 때문에 우리는 1 스코어를 얻게 될 것이다. 그리고 총 Loss는 C-1을 얻게 되는 것이다. 이는 실제로 '디버깅 전략'으로 유용하다. 이런 전략을 가지고 트레이닝을 처음 시작할 때 Loss가 C-1이 아니라면 아마 버그가 있는 것이고 고쳐야 할 것이다.

```python
def L_i_vectorized(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```



# 2. 규제(Regularization)

<img src='Image/cs231n010.png' width='100%'>

* 모델이 더 복잡해지지 못하도록
* 모델에 soft penalty를 추가하도록
* "만약 너가 복잡한 모델을 계속 쓰고 싶으면, 이 penalty를 감수해야 할 거야!"

<img src='Image/cs231n011.png' width='100%'>

* L1: 분류기의 복잡도를가중치 W의 0의 개수에 따라 측정한다.
* L2: 분류기의 복잡도를 상대적으로 더 coarse한지(값이 매끄러운지)를 통해 측정한다.

39:21~