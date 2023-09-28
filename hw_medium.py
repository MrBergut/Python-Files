from tensorflow.keras.datasets import mnist
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
from tensorflow.keras.preprocessing.image import load_img
# Подключение библиотеки для работы с массивами
import numpy as np
# Подключение библиотек для отрисовки изображений
import matplotlib.pyplot as plt
# Подключение модуля для работы с файлами
import os
from PIL import Image
import pylab

(x_train_org, y_train_org), (x_test_org, y_test_org) = mnist.load_data()

# n = 12
# plt.imshow(Image.fromarray(x_train_org[n]).convert('RGBA'))
# plt.show()

# Изменение формата из матричного в векторный
x_train = x_train_org.reshape(60000, 784)
x_test = x_test_org.reshape(10000, 784)

# print(x_train.shape)
# print(x_test.shape)

x_train = x_train.astype('float32')
# деление на 255 для диапозона от 0 до 1
x_train = x_train / 255

x_test = x_test.astype('float32')
# деление на 255 для диапозона от 0 до 1
x_test = x_test / 255

y_train = utils.to_categorical(y_train_org, 10)
y_test = utils.to_categorical(y_test_org, 10)

model = Sequential()
model.add(Dense(800, input_dim=784, activation = 'relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


model.fit(x_train, y_train, batch_size = 128, epochs = 15, verbose = 1)

# import gdown
# print(gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l3/hw_upro.zip', None, quiet=True))

# Вывод для примера картинок по классу
# Создание полотна из 10 графиков
fig, axs = plt.subplots(1, 10, figsize=(25, 5))
# Проход и отрисовка по всем классам
for i in range(10):
    img = load_img('digits/' + str(i) + '.png', target_size = (28, 28), color_mode = 'grayscale')
    img = np.array(1 - np.array(img) / 255)
    img[img < 0.5] = 0
    img[img > 0.5] = 1
    axs[i].imshow(img, cmap='gray')
    # Вывод распознования
    result = 'НЕВЕРНО' if (i != np.argmax(model.predict(img.reshape(1, 784)))) else 'Верно'
    print('число', i, '.сеть распознала', np.argmax(model.predict(img.reshape(1, 784))), result)


plt.show()

# Вывод для примера по каждому классу
