from pycaret.datasets import get_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data=pd.read_csv("d:\\data_science_24\project\data_12f.csv", sep=';')
data_kl=pd.read_csv("d:\\data_science_24\project\data_12f_kl.csv", sep=';')

from pycaret.classification import *
s = setup(data, target = 'TARGET', session_id = 123)
# functional API

best = compare_models()
print(best)
evaluate_model(best)
rf = create_model('rf')
#plot_model(rf, plot = 'auc')
# functional API
#plot_model(rf, plot = 'confusion_matrix')
plot_model(rf, plot = 'class_report')
predict_model(best)
predictions = predict_model(best, data=data)
print(predictions.head())
# сохранение модели
# functional API
save_model(best, 'my_best_pipeline')
