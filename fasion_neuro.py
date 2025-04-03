from tensorflow.keras.datasets import fashion_mnist # загрузка dataset MNIST
from tensorflow.keras.models import Sequential # указываем на последовательную нейронную сеть
from tensorflow.keras.layers import Dense # указывает тип слоёв
from tensorflow.keras import utils # подключаем необходимые утилиты keras для преобразования данных
from kerastuner import RandomSearch, Hyperband, BayesianOptimization # подключение модулей Keras Tuner
import numpy as np

# Загрузка данных
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Преобразование размерности изображений
x_train = x_train.reshape(60000, 784).astype('float32')

# Нормализация данных изображения
x_train /= 255

# Преобразование меток в категории
y_train = utils.to_categorical(y_train, 10)

# Присвоение названия классам
classes = ['футболка', 'брюки', 'свитер', 'платье',
           'пальто', 'туфли', 'рубашка', 'кроссовки',
           'сумка', 'ботинки']

# Создаём модель
model = Sequential()

# Добавление уровней сети
model.add(Dense(800, input_dim = 784, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

# Компиляция модели
model.compile(loss = "categorical_crossentropy",
              optimizer = "SGD", metrics = ["accuracy"])

print(model.summary())

# Обучение сети
model.fit(x_train,
          y_train,
          batch_size = 200,
          epochs = 100,
          validation_split = 0.2,
          verbose = 1)

# Сохранение нейронной сети
model.save('fasion_ai.h5')

# Преобразование тестовых данных
x_test = x_test.reshape(10000, 784).astype('float32')

# Нормализация тестовой выборки
x_test /= 255

# Преобразованине меток в категории
y_test = utils.to_categorical(y_test, 10)

# Оценка качества обучения сети на тестовых данных
scores = model.evaluate(x_test, y_test, verbose = 1)

mark = round((scores[1] * 100), 4)

print(f"Доля верных ответов на тестовых данных (%): {mark}")