import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. MNIST 데이터셋 불러오기
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 데이터 전처리
x = x.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # 학습 데이터 정규화
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0  # 테스트 데이터 정규화

# 라벨 원-핫 인코딩
y = tf.keras.utils.to_categorical(y, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 학습 데이터와 검증 데이터 분리
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. 데이터 증강
datagen = ImageDataGenerator(
    rotation_range=10,       # 회전 범위
    width_shift_range=0.1,   # 가로 이동
    height_shift_range=0.1,  # 세로 이동
    zoom_range=0.1           # 줌
)
datagen.fit(x_train)

# 4. CNN 모델 정의
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),  # 과적합 방지를 위한 Dropout
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# 5. 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. 콜백 정의
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # 조기 종료
    ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss'),  # 최적 모델 저장
    TensorBoard(log_dir='./logs')  # TensorBoard 로그 저장
]

# 7. 모델 학습
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=50,
    callbacks=callbacks,
    steps_per_epoch=len(x_train) // 32
)

# 8. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc}")

# 9. 학습 결과 시각화
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

# 10. 예측 결과 시각화
def visualize_prediction(model, x_test, y_test):
    y_pred = model.predict(x_test)
    for i in range(5):  # 샘플 5개 출력
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Predicted: {np.argmax(y_pred[i])}, True: {np.argmax(y_test[i])}")
        plt.axis('off')
        plt.show()

visualize_prediction(model, x_test, y_test)
