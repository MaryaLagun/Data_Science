from pycaret.datasets import get_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv("d:\\data_science_24\project\data_12f.csv", sep=';')
data_kl=pd.read_csv("d:\\data_science_24\project\data_12f_kl.csv", sep=';')
from pycaret.clustering import *
s = setup(data_kl, normalize = True)
# Создание модели
# functional API
kmeans = create_model('kmeans')
print(kmeans)
#evaluate_model(kmeans)
#plot_model(kmeans, plot = 'elbow')
plot_model(kmeans, plot = 'silhouette')
# functional API
result = assign_model(kmeans)
print(result.head())
predictions = predict_model(kmeans, data = data_kl)
predictions.head()
save_model(kmeans, 'kmeans_pipeline')