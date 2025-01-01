## 1. Import Packages
사용할 패키지를 가져옵니다.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
```

## 2. Training and Testing Data Loading
학습과 테스트에 필요한 데이터를 불러옵니다.

```python
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

<p align="center">Data Set Preview</p>

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png" alt="MNIST 이미지">
</p>

<p align="center">
  mnist image from <a href="https://commons.wikimedia.org/wiki/File:MnistExamples.png">wikimedia</a>
</p>

## 3. Reshape Images

```python
x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

```
