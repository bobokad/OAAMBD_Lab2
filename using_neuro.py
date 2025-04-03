from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import numpy as np
import os

# Загрузка нейросети из файла
model = load_model('best_model_1.h5')

# Список классов Fashion-MNIST
classes = ['футболка', 'брюки', 'свитер', 'платье',
           'пальто', 'туфли', 'рубашка', 'кроссовки',
           'сумка', 'ботинки']

# Функция предобработки изображения под формат Fashion-MNIST
def prepare_and_save_image(path, save_path):
    # 1. Открытие изображения и перевод в оттенки серого
    img = Image.open(path).convert('L')  # grayscale

    # 2. Инвертирование цветов (фон -> чёрный, объект -> белый)
    img = ImageOps.invert(img)

    # 3. Пороговая бинаризация
    img = img.point(lambda x: 0 if x < 30 else 255, '1')  # можно подстроить порог

    # 4. Конвертация обратно в 'L' и обрезка по границам
    img = img.convert('L')
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # 5. Масштабирование объекта до 20x20
    img = img.resize((20, 20), Image.BICUBIC)

    # 6. Центрирование объекта на чёрном фоне 28x28
    new_img = Image.new('L', (28, 28), 0)
    new_img.paste(img, ((28 - 20) // 2, (28 - 20) // 2))

    # 7. Сохранение изображения для визуальной проверки
    new_img.save(save_path)

    # 8. Преобразование в numpy-массив и нормализация
    img_array = np.array(new_img).reshape(1, 784).astype('float32') / 255
    return img_array

# Список обрабатываемых изображений
image_files = ['bag.jpg', 'dress.jpg', 'shirt.jpg', 'keds.jpg', 'tro.jpg']

# Обработка, сохранение и предсказание для каждого изображения
for file_name in image_files:
    try:
        processed_name = f"processed_{file_name}"
        img_array = prepare_and_save_image(file_name, processed_name)

        # Предсказание с помощью модели
        prediction = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction)

        print(f"Фото: {file_name} → {classes[predicted_class]}")
        

    except Exception as e:
        print(f"Ошибка при обработке {file_name}: {e}")