from tensorflow.keras.datasets import fashion_mnist # загрузка dataset MNIST
from tensorflow.keras.models import Sequential # указываем на последовательную нейронную сеть
from tensorflow.keras.layers import Dense # указывает тип слоёв
from tensorflow.keras import utils, layers # подключаем необходимые утилиты keras для преобразования данных
from tensorflow.keras.layers import Dropout # подключаем слой регуляризации Dropout
from tensorflow.keras.layers import BatchNormalization # подключаем слой нормализации
from tensorflow.keras.callbacks import EarlyStopping # подключаем раннюю остановку обучения
from tensorflow.keras import optimizers # доступ к разным оптимизаторам и learning_rate
from kerastuner import RandomSearch, Hyperband, BayesianOptimization # подключение модулей Keras Tuner
import numpy as np


# Функция создания модели нейронной сети
def build_model(hp):
    model = Sequential()

    # Гиперпараметры активации и оптимизатора
    activation_choice = hp.Choice('activation', values = ['relu', 'sigmoid',
                                                          'tanh', 'elu', 'selu'])
    optimizer_choice = hp.Choice('optimizer', values = ['adam', 'rmsprop', 'SGD'])

    # Гиперпараметр learning_rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    # Первый (входной) слой
    model.add(Dense(units = hp.Int('units_input',     # полносвязный слой с разным количеством нейронов
                                     min_value = 512,   # минимальное количество нейронов
                                     max_value = 1024,  # максимальное количество нейронов
                                     step = 32),
                    input_dim = 784,
                    activation = activation_choice))
    model.add(BatchNormalization())  # нормализация активаций
    model.add(Dropout(hp.Float('dropout_input', min_value=0.2, max_value=0.5, step=0.1))) # Dropout после входного слоя

    # Цикл добавления скрытых слоёв
    for i in range(hp.Int('num_layers', 2, 5)): # количество скрытых слоёв
        model.add(layers.Dense(units = hp.Int('units_' + str(i), # количество нейронов в каждом слое
                                              min_value = 128,
                                              max_value = 1024,
                                              step = 32),
                            activation = activation_choice))
        model.add(BatchNormalization())  # нормализация активаций после каждого скрытого слоя
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1))) # Dropout

    # Выходной слой
    model.add(Dense(10, activation = 'softmax'))

    # Инициализация оптимизатора с подобранным learning_rate
    if optimizer_choice == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = optimizers.SGD(learning_rate=learning_rate)

    # Компиляция модели
    model.compile(
        optimizer = optimizer,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])
    
    return model

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

# Преобразование тестовых данных
x_test = x_test.reshape(10000, 784).astype('float32')

# Нормализация тестовой выборки
x_test /= 255

# Преобразование меток в категории
y_test = utils.to_categorical(y_test, 10)

# Реализация тюнера BayesianOptimization
tuner = BayesianOptimization(
    build_model,    # функция создания модели
    objective = 'val_accuracy', # метрика, которую необходимо оптимизировать
                                # доля правильных ответов на проверочном наборе данных
    max_trials = 30,   # количество запусков обучения
    directory = 'test_directory' # каталог сохранения обученной модели
)

# Пространство поиска подбора гиперпараметров
tuner.search_space_summary()

# Ранняя остановка при отсутствии улучшения
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Подбор гиперпараметров
tuner.search(x_train,   # Данные для обучения
             y_train,   # Метки правильных ответов
             batch_size = 256,  # Размер мини-выборки
             epochs = 50,  # Количество эпох обучения
             validation_split = 0.2,    # Процентное число данных, которое будет использоваться для проверки
             callbacks=[early_stop],    # Ранняя остановка
             verbose = 1)

# Выбор наилучшей модели
tuner.results_summary()

# Получение трёх наилучших моделей
models = tuner.get_best_models(num_models = 3)

# Сохранение и оценка каждой модели на тестовых данных
for i, model in enumerate(models):
    print(f"\nМодель №{i + 1}")
    model.summary()
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Точность: {acc * 100:.2f}%")

    # Сохраняем модель в формате .h5
    model.save(f"best_model_{i + 1}.h5")
    print(f"Модель сохранена как best_model_{i + 1}.h5")