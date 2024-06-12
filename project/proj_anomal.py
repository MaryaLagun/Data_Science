from pycaret.datasets import get_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv("d:\\data_science_24\project\data_12f.csv", sep=';')
data_kl=pd.read_csv("d:\\data_science_24\project\data_12f_kl.csv", sep=';')
from pycaret.anomaly import *
s = setup(data_kl, session_id = 123)
# Создание модели
# functional API
iforest = create_model('iforest')
print(iforest)
models()
# Анализ модели
# functional API
plot_model(iforest, plot = 'tsne')
# Назначить модель
# functional API
result = assign_model(iforest)
result.head()
# Прогноз
# functional API
predictions = predict_model(iforest, data = data_kl)
predictions.head()
# Сохранение модели
# functional API
save_model(iforest, 'iforest_pipeline')