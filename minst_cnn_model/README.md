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
MNIST 데이터셋은 (샘플 개수, 784) 형태의 1D 배열로 제공됩니다. 이를 CNN 에 전달하기 위해서는 (높이, 너비, 채널) 형태의 2D 배열로 변환하여합니다.  
-1 은 자동으로 샘플의 개수를 맞추라는 의미입니다. ex) 샘플이 60,000개라면 reshape 은 (60000, 28, 28, 1) 로 변환합니다.  
MNIST 데이터셋은 정수형(int8)로 제공됩니다. CNN 모델의 연산은 실수(float)로 수행되어 .astype('float32')로 변환합니다.  
MNIST 데이터셋의 픽셀 값은 [0, 255] 범위의 정수로 표현됩니다. 딥러닝 모델의 학습은 입력 데이터의 값 범위가 작을때 더 효과적이므로 
/ 255.0 으로 [0, 1] 범위로 정규화 합니다.

```python
x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

```

### 4. One-Hot Encoding
숫자 레이블(ex: 0, 1, 2)은 연속적인 값처럼 보이지만, 사실 클래스 사이에 순서나 크기 비교가 없습니다. (ex: 2가 1보다 크다는 의미가 없습니다.)  
원-핫 인코딩은 이 순서/크기 정보의 왜곡을 제거하고 각 클래스가 독립적임을 모델이 이해할 수 있도록 도와줍니다.

```python
y = tf.keras.utils.to_categorical(y, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

### 5. 
