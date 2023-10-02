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
# Подключение библиотеки для работы с массивами
import numpy as np
# Подключение библиотек для отрисовки изображений
import matplotlib.pyplot as plt
# Подключение модуля для работы с файлами
import os
from PIL import Image
from sklearn.model_selection import train_test_split

(x, y), (__, __) = mnist.load_data()

# model.fit(x_train, y_train, batch_size = 8, epochs = 100, validation_spirit = 0.2, verbose = 1)
# model.fit(x_train, y_train, batch_size = 8, epochs = 100, validation_data = (x_train[56000:]), verbose = 1
# 
# )




# print(x.shape)
# print(y.shape)

# x_train = x_train[:48000]
# x_val = x_train[48000:54000]
# x_test = x_train[54000:]

# y_train = y_train[:48000]
# y_val = y_train[48000:54000]
# y_test = y_train[54000:]

# print("Обучающая выборка", x_trayn.shape)
# print('Валидационная выборка', x_val.shape)
# print('Тестовая выборка ', x_test.shape)

# print("Обучающая выборка", y_trayn.shape)
# print('Валидационная выборка', y_val.shape)
# print('Тестовая выборка ', y_test.shape)

# valMask
# splitVal = 0.2

# mask_list = np.random.sample(x.shape[0])

# valMask = mask_list < splitVal

# tr = 0
# fl = 0
# for i in valMask:
#     if i:
#         tr =+ 1
#     else:
#         fl =+ 1

# print('колличество true', tr)
# print('колличество false', fl)
# print()
# print('заданный процент разделения', splitVal)
# print(valMask)
# print(~valMask)
# print('Доля true', round(tr/(tr+fl), 2))
# print('Доля false', round(fl/(tr+fl), 2))



# def unison_shuffled_split(a,
#                           b,
#                           test_size
#                           ):
#     a = np.array(a)
#     b = np.array(b)

#     assert len(a) == len(b)

#     p = np.random.permutation(len(a))
#     s = round(len(a) * test_size) 

#     a = a[p].tolist() # перемещение элементов согласно перемешанному списку индексов
#     b = b[p].tolist()

#     return a[s:], a[:s], b[s:], b[:s]

# x_train, x_test, y_train, y_test = unison_shuffled_split(x, #датасет изображения
#                                                          y, # датасет метки
#                                                          0.1, # процент тестовых значений
#                                                          )

# print(len(x_train))
# print(len(y_train))
# print('')
# print(len(x_test))
# print(len(y_test))