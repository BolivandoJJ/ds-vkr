from tensorflow import keras
import pickle
import numpy as np

ns_model = keras.models.load_model('./model')
file = open('./normalizator_pickles/input_normalizator.pickle', 'rb')
input_normalizer = pickle.load(file)
file.close()
file = open('./normalizator_pickles/output_normalizator.pickle', 'rb')
output_normalizer = pickle.load(file)
file.close()
input_data = [float(input('Введите: Плотность, кг/м3 ')),
              float(input('Введите: Модуль упругости, ГПа ')),
              float(input('Введите: Количество отвердителя, м.% ')),
              float(input('Введите: Содержание эпоксидных групп, %_2 ')),
              float(input('Введите: Температура вспышки, С_2 ')),
              float(input('Введите: Поверхностная плотность, г/м2 ')),
              float(input('Введите: Потребление смолы, г/м2 ')),
              float(input('Введите: Угол нашивки, град ')),
              float(input('Введите: Шаг нашивки ')),
              float(input('Введите: Плотность нашивки '))]
input_data_normalized = input_normalizer.transform([input_data])
output_data_normalized = ns_model.predict([input_data_normalized])
output_data = output_normalizer.inverse_transform(output_data_normalized)
print('Модуль упругости при растяжении, ГПа: ' + str(output_data[0][0]))
print('Соотношение матрица-наполнитель: ' + str(output_data[0][1]))
print('Прочность при растяжении, МПа: ' + str(output_data[0][2]))