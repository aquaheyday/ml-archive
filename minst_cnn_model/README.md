# 미리보기
url : https://aquaheyday.github.io/web/mnist_cnn_model

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

## 3. 데이터 전처리(Data Preprocessing)

### 3-1. Reshape Images
MNIST 데이터셋은 (샘플 개수, 784) 형태의 1D 배열로 제공됩니다. 이를 CNN 에 전달하기 위해서는 (높이, 너비, 채널) 형태의 2D 배열로 변환하여합니다.  
-1 은 자동으로 샘플의 개수를 맞추라는 의미입니다. ex) 샘플이 60,000개라면 reshape 은 (60000, 28, 28, 1) 로 변환합니다.  
MNIST 데이터셋은 정수형(int8)로 제공됩니다. CNN 모델의 연산은 실수(float)로 수행되어 .astype('float32')로 변환합니다.  
MNIST 데이터셋의 픽셀 값은 [0, 255] 범위의 정수로 표현됩니다. 딥러닝 모델의 학습은 입력 데이터의 값 범위가 작을때 더 효과적이므로 
/ 255.0 으로 [0, 1] 범위로 정규화 합니다.

```python
x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

```

### 3-2. One-Hot Encoding
숫자 레이블(ex: 0, 1, 2)은 연속적인 값처럼 보이지만, 사실 클래스 사이에 순서나 크기 비교가 없습니다. (ex: 2가 1보다 크다는 의미가 없습니다.)  
원-핫 인코딩은 이 순서/크기 정보의 왜곡을 제거하고 각 클래스가 독립적임을 모델이 이해할 수 있도록 도와줍니다.

```python
y = tf.keras.utils.to_categorical(y, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

### 3-3. The separation of training data and validation data
학습 데이터와 검증 데이터 분리

```python
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
```

### 4. 데이터 증강(Data augmentation)
데이터 증강은 데이터를 더 풍부하게 만들어 모델이 다양한 학습할 수 있게 합니다.

```python
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)
```
rotation_range=10: 이미지의 회전 범위를 -10도에서 +10도 사이로 지정합니다.  
width_shift_range=0.1: 이미지의 가로 방향으로 10% 범위에서 임의로 이동합니다.  
height_shift_range=0.1: 이미지의 세로 방향으로 10% 범위에서 임의로 이동합니다.  
zoom_range=0.1: 확대/축소 비율은 90%~110% 사이에서 무작위로 적용됩니다.  

### 5. CNN 모델 정의

```python
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

### 6. 모델 컴파일

```python
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
```

### 7. 콜백 정의

```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss'),
    TensorBoard(log_dir='./logs')
]
```

### 8. 모델 학습

```python
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=callbacks,
    steps_per_epoch=len(x_train) // 32
)
```

### 9. 모델 평가

```python
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")
```

### 10. 학습 결과 시각화

```python
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # 정확도
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    # 손실
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.show()

plot_training_history(history)
```

### 11. 예측 결과 시각화

```python
def visualize_prediction(model, x_test, y_test):
    y_pred = model.predict(x_test)
    for i in range(5):  # 샘플 5개 출력
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {np.argmax(y_pred[i])}, True: {np.argmax(y_test[i])}")
        plt.axis('off')
        plt.show()

visualize_prediction(model, x_test, y_test)
```
