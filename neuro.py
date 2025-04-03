import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Загружаем набор данных Fashion MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Нормализация данных (приведение значений пикселей в диапазон [-0.5, 0.5])
x_train = x_train.astype(np.float32) / 255.0 - 0.5
x_test = x_test.astype(np.float32) / 255.0 - 0.5

# Преобразование изображений в одномерные векторы
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Преобразуем метки классов в one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Создаем модель многослойного персептрона
model = keras.Sequential([
    layers.Dense(128, activation='elu', input_shape=(784,)),
    layers.Dense(128, activation='elu'),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели с функцией потерь и оптимизатором Adam
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Обучение модели
model.fit(
    x_train, y_train,
    batch_size=64,  # Размер мини-батча
    epochs=10,      # Количество эпох
    validation_data=(x_test, y_test)
)

# Оценка точности модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Точность модели на тестовых данных: {test_acc:.4f}")