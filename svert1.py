# Работа с массивами
import numpy as np

# Генератор аугментированных изображений
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Основа для создания последовательной модели
from tensorflow.keras.models import Sequential
# Основные слои
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
# Оптимизатор
from tensorflow.keras.optimizers import Adam
# Матрица ошибок классификатора
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Подключение модуля для загрузки данных из облака
import gdown
# Инструменты для работы с файлами
import os
# Отрисовка графиков
import matplotlib.pyplot as plt

# Задание гиперпараметров

TRAIN_PATH          = 'cars'       # Папка для обучающего набора данных
TEST_PATH           = 'cars_test'  # Папка для тестового набора данных

TEST_SPLIT          = 0.1                   # Доля тестовых данных в общем наборе
VAL_SPLIT           = 0.2                   # Доля проверочной выборки в обучающем наборе

IMG_WIDTH           = 128                   # Ширина изображения для нейросети
IMG_HEIGHT          = 64                    # Высота изображения для нейросети
IMG_CHANNELS        = 3                     # Количество каналов (для RGB равно 3, для Grey равно 1)

# Параметры аугментации
ROTATION_RANGE      = 8                     # Пределы поворота
WIDTH_SHIFT_RANGE   = 0.15                  # Пределы сдвига по горизонтали
HEIGHT_SHIFT_RANGE  = 0.15                  # Пределы сдвига по вертикали
ZOOM_RANGE          = 0.15                  # Пределы увеличения/уменьшения
BRIGHTNESS_RANGE    = (0.7, 1.3)            # Пределы изменения яркости
HORIZONTAL_FLIP     = True                  # Горизонтальное отражение разрешено

EPOCHS              = 20                    # Число эпох обучения
BATCH_SIZE          = 24                    # Размер батча для обучения модели
OPTIMIZER           = Adam(0.0001)          # Оптимизатор


# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(TRAIN_PATH))

# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)

# Проверка результата
# print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')

int(1088 * 0.1)

photo_1 = 'cars/Ferrari/car_Ferrari__0.png'
photo_1 = 'cars_test/Ferrari/car_Ferrari__0.png'

# Перенос файлов для теста в отдельное дерево папок, расчет размеров наборов данных

try:
  os.mkdir(TEST_PATH)                                        # Создание папки для тестовых данных
except:
  pass

train_count = 0
test_count = 0

for class_name in CLASS_LIST:                              # Для всех классов по порядку номеров (их меток)
    class_path = f'{TRAIN_PATH}/{class_name}'              # Формирование полного пути к папке с изображениями класса
    test_path = f'{TEST_PATH}/{class_name}'                # Полный путь для тестовых данных класса
    class_files = os.listdir(class_path)                   # Получение списка имен файлов с изображениями текущего класса
    class_file_count = len(class_files)                    # Получение общего числа файлов класса

    try:
      os.mkdir(test_path)                                    # Создание подпапки класса для тестовых данных
    except:
      pass

    test_file_count = int(class_file_count * TEST_SPLIT)   # Определение числа тестовых файлов для класса
    test_files = class_files[-test_file_count:]            # Выделение файлов для теста от конца списка
    for f in test_files:                                   # Перемещение тестовых файлов в папку для теста
        os.rename(f'{class_path}/{f}', f'{test_path}/{f}')
    train_count += class_file_count                        # Увеличение общего счетчика файлов обучающего набора
    test_count += test_file_count                          # Увеличение общего счетчика файлов тестового набора

    print(f'Размер класса {class_name}: {class_file_count} машин, для теста выделено файлов: {test_file_count}')

print(f'Общий размер базы: {train_count}, выделено для обучения: {train_count - test_count}, для теста: {test_count}')

# Генераторы изображений

# Изображения для обучающего набора нормализуются и аугментируются согласно заданным гиперпараметрам
# Далее набор будет разделен на обучающую и проверочную выборку в соотношении VAL_SPLIT
train_datagen = ImageDataGenerator(
                    rescale=1. / 255.,
                    rotation_range=ROTATION_RANGE,
                    width_shift_range=WIDTH_SHIFT_RANGE,
                    height_shift_range=HEIGHT_SHIFT_RANGE,
                    zoom_range=ZOOM_RANGE,
                    brightness_range=BRIGHTNESS_RANGE,
                    horizontal_flip=HORIZONTAL_FLIP,
                    validation_split=VAL_SPLIT
                )

# Изображения для тестового набора только нормализуются
test_datagen = ImageDataGenerator(
                   rescale=1. / 255.
                )

# Обучающая выборка генерируется из папки обучающего набора
train_generator = train_datagen.flow_from_directory(
    # Путь к обучающим изображениям
    TRAIN_PATH,
    # Параметры требуемого размера изображения
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    # Размер батча
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    # Указание сгенерировать обучающую выборку
    subset='training'
)

# Проверочная выборка также генерируется из папки обучающего набора
validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    # Указание сгенерировать проверочную выборку
    subset='validation'
)

# Тестовая выборка генерируется из папки тестового набора
test_generator = test_datagen.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=test_count,
    class_mode='categorical',
    shuffle=True,
)
#Проверка формы данных
# print(f'Форма данных тренировочной выборки: {train_generator[0][0].shape},
#        {train_generator[0][1].shape}, батчей: {len(train_generator)}')

# print(f'Форма данных проверочной выборки: {validation_generator[0][0].shape},
#        {validation_generator[0][1].shape}, батчей: {len(validation_generator)}')

# print(f'Форма данных тестовой выборки: {test_generator[0][0].shape},
#        {test_generator[0][1].shape}, батчей: {len(test_generator)}')

# print()
# #Проверка назначения меток класса
# print(f'Метки классов тренировочной выборки: {train_generator.class_indices}')
# print(f'Метки классов проверочной выборки: {validation_generator.class_indices}')
# print(f'Метки классов тестовой выборки: {test_generator.class_indices}')

# plt.imshow(train_generator[1][0][2])
# plt.show()

#Функция рисования образцов изображений из заданной выборки
def show_batch(batch, #батч с примерами
               img_range = range(20),  # диапозон номеров картинок
               figsize = (25, 8), # размер полотна для рисовования 1 строки таблицы
               columns = 5
               ): # число колонок в таблице
   for i in img_range:
      ix = i % columns
      if ix == 0:
         fig, ax = plt.subplots(1, columns, figsize = figsize)
      class_label = np.argmax(batch[1][i])
      ax[ix].set_title(CLASS_LIST[class_label])
      ax[ix].imshow(batch[0][i])
      ax[ix].axis('off')
      plt.tight_layout()
   plt.show()
show_batch(test_generator[0])

def compile_train_model(model, # модель НС
                        train_data, # обучающие данные
                        val_data, # проверочные данные
                        optimizer = OPTIMIZER, # оптимизатор
                        epochs = EPOCHS, # количество эпох обучения
                        batch_size = BATCH_SIZE, # размер батча
                        figsize = (20, 5) # размер полотна для графиков
                        ):
    # Компиляция модели
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

    #Вывод сводки
    model.summary()

    # Обучение модели с заданными параметрами
    history = model.fit(train_data, epochs = epochs, 
                        #batch_size = batch_size,
                        validation_data = val_data
                          )
    #Вывод графиков точности и ошибок
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize)
    fig.suptitle('График процесса обучения модели')
    ax1.plot(history.history['accuracy'],
             label='Доля верных ответов на обучающем наборе')
    ax1.plot(history.history['val_accuracy'],
             label='Доля верных ответов на тестовом наборе')
    ax1.xaxis.get_major_locator().set_params(integer = True)
    ax1.set_xlabel('Эпохи обучения')
    ax1.set_ylabel('Доля верных ответов')
    ax1.legend()

    ax2.plot(history.history['loss'],
             label='Ошибка на обучающем наборе')
    ax2.plot(history.history['val_loss'],
             label='Ошибка на тестовом наборе')
    ax2.xaxis.get_major_locator().set_params(integer = True)
    ax2.set_xlabel('Эпохи обучения')
    ax2.set_ylabel('Ошибка')
    ax2.legend()

def eval_model(model,
               x,
               y_true,
               class_labels = [],
               cm_round = 3,
               title = '',
               figsize = (10, 10)
               ):
    y_pred = model.predict(x)
    cm = confusion_matrix(np.argmax(y_true, axis=1),
                          np.argmax(y_pred, axis=1),
                          normalize='true')

    cm = np.arount(cm, cm_round)


    fig, ax = plt.subplots(figsize = figsize)
    ax.set_title(f'Нейросеть {title}: матрица ошибок нормализированная', fontsize = 18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = class_labels)
    disp.plot(ax=ax)
    ax.images[-1].colorbar.remove()
    fig.autofmt_xdate(rotation=45)
    plt.xlabel('Предсказанные классы', fontsize = 16)
    plt.ylabel('Верные классы',fontsize = 16)

    print('-' * 100)
    print(f'нейросеть: {title}')

    for cls in range(len(class_labels)):
        cls_pred = np.argmax(cm[cls])
        msg = 'Верно 😀' if cls_pred == cls else 'Неверно 😔'

        print('Класс {:<20} {:3.0f}% сеть отнесла к классу {:<20} - {}'.format(class_labels[cls],
                                                                               100. * cm[cls, cls_pred],
                                                                               class_labels[cls_pred],
                                                                               msg))

    print('\nСредняя точность распознования: {:3.0}%'.format(100. * cm.diagonal().mean()))

def compile_train_eval_model(model,
                             train_data,
                             val_data,
                             test_data,
                             class_labels = CLASS_LIST,
                             title = '',
                             optimizer = OPTIMIZER,
                             epochs = EPOCHS,
                             batch_size = BATCH_SIZE,
                             graph_size = (20, 5),
                             cm_size = (10, 10)
                             ):
    compile_train_model(model, train_data, val_data, optimizer=optimizer, epochs=epochs, batch_size=batch_size, figsize=graph_size)
    eval_model(model, test_data[0][0], test_data[0][1],
               class_labels = class_labels, title=title, figsize=cm_size)


model_conv = Sequential()

model_conv.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
model_conv.add(BatchNormalization())

model_conv.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model_conv.add(MaxPooling2D(pool_size=(3, 3)))

model_conv.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model_conv.add(BatchNormalization())
model_conv.add(Dropout(0.2))

model_conv.add(Conv2D(256, (3, 3), padding = 'same', activation = 'relu'))
model_conv.add(MaxPooling2D(pool_size=(3,3)))
model_conv.add(Dropout(0.2))

model_conv.add(Conv2D(512, (3, 3), padding = 'same', activation = 'relu'))
model_conv.add(BatchNormalization())

model_conv.add(Conv2D(1024, (3, 3), padding = 'same', activation = 'relu'))
model_conv.add(MaxPooling2D(pool_size=(3, 3)))
model_conv.add(Dropout(0.2))

model_conv.add(Flatten())

model_conv.add(Dense(2048, activation='relu'))

model_conv.add(Dense(4096, activation='relu'))

model_conv.add(Dense(CLASS_COUNT, activation='softmax'))

compile_train_eval_model(model_conv, train_generator, validation_generator, test_generator, class_labels = CLASS_LIST, title = 'Сверточный классификатор')