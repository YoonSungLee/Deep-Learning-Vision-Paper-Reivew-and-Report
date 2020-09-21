# U-Net: Convolutional Networks

---

<img src='https://i.imgur.com/33PEWdw.png' widtth='100%'>

# 1. Summary
* Symmetric U-shaped network built upon FCN(fully convolutional network) for image segmentation
* Contracting path to capture context, expanding path that enables precise localization
* concatenate that provides local information to the global information while upsampling
* Data augmentation that allows the network to learn invariance to deformations with very little training data

# 2. Contracting Path
<img src='https://i.imgur.com/7Oc4NO5.png' width='100%'>

# 3. Expanding Path
<img src='https://i.imgur.com/yZrumeF.png' width='100%'>

# 4. Skip Architecture
<img src='https://i.imgur.com/2TlxjLo.png' width='100%'>

# Reference
* https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a