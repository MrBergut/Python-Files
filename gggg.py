# Подключение класса для создания нейронной сети прямого распространения
from tensorflow.keras.models import Sequential
# Подключение класса для создания полносвязного слоя
from tensorflow.keras.layers import Dense
# Подключение оптимизатора
from tensorflow.keras.optimizers import Adam
# Подключение утилит для to_categorical
from tensorflow.keras import utils
# Подключение библиотеки для загрузки изображений
from tensorflow.keras.preprocessing import image
# Подключение библиотеки для работы с массивами
import numpy as np
# Подключение библиотек для отрисовки изображений
import matplotlib.pyplot as plt
# Подключение модуля для работы с файлами
import os
from PIL import Image

# import gdown
# print(gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l3/hw_pro.zip', None, quiet=True))

# Путь к директории с базой
base_dir = 'hw_pro'
# Создание пустого списка для загрузки изображений обучающей выборки
x_train = []
# Создание списка для меток классов
y_train = []
# Задание высоты и ширины загружаемых изображений
img_height = 20
img_width = 20
# Перебор папок в директории базы
for patch in os.listdir(base_dir):
    # Перебор файлов в папках
    for img in os.listdir(base_dir + '/' + patch):
        # Добавление в список изображений текущей картинки
        x_train.append(image.img_to_array(image.load_img(base_dir + '/' + patch + '/' + img,
                                                    target_size=(img_height, img_width),
                                                    color_mode='grayscale')))
        # Добавление в массив меток, соответствующих классам
        if patch == '0':
            y_train.append(0)
        else:
            y_train.append(1)

# Преобразование в numpy-массив загруженных изображений и меток классов
x_train = np.array(x_train)
y_train = np.array(y_train)
# Вывод размерностей
# print('Размер массива x_train', x_train.shape)
# print('Размер массива y_train', y_train.shape)

# Вывод примера изображения из базы
# n_example = 5
# plt.imshow(np.reshape(x_train[2], (20,20)), cmap = 'gray')
# plt.show()
# print(y_train[n_example])

x_train = x_train.reshape(x_train.shape[0], 400)
#Нормирование входных картинок
#Преобразование x_train в числа с плавающей точкой
x_train = x_train.astype('float32')
x_train = x_train / 255

y_train = utils.to_categorical(y_train, 2)

#Список сохранения точности сети при загрузке параметров
data_list = []
#Перебор значений в списке с кличеством нейронов

model = Sequential()
model.add(Dense(1000, input_dim= 400, activation = 'relu'))
# Создание полносвязного слоя
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
#Вывод параметров текущей сети

history = model.fit(x_train, y_train, batch_size = 10, epochs = 10, verbose = 1, shuffle = True)


