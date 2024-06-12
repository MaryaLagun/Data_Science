import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import oracledb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn import tree
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("d:\\data_science_24\project\data_12f.csv", sep=';')
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

# Предобработка данных
# Кодирование категориальных признаков
label_encoders = {}
for column in ['OKED']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Определение признаков и целевой переменной
X = df[['OKED', 'G01', 'G03', 'G07', 'G11','G77','CHISL01','CHISL02']]
y = df['TARGET']

# Кодирование целевой переменной
le_disease = LabelEncoder()
y = le_disease.fit_transform(y)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение классификатора Gradient Boosting
gb_classifier = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
gb_classifier.fit(X_train, y_train)

# Прогнозирование классов на тестовом наборе данных
y_pred = gb_classifier.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy GradientBoostingClassifier: {accuracy:.2f}')
print('Classification Report GradientBoostingClassifier:')
print(report)


# Dummy Classifier с параметром most_frequent
# Импорт библиотек

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Создание и обучение Dummy Classifier
dummy_clf = DummyClassifier(strategy="most_frequent") # stratified
dummy_clf.fit(X_train, y_train)

# Оценка модели
y_pred = dummy_clf.predict(X_test)
print("Dummy Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Dummy Classifier Classification Report:")
#print(classification_report(y_test, y_pred))
print("Dummy Classifier Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение модели
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание и оценка
y_pred = model.predict(X_test)
print(f"Accuracy RandomForest: {accuracy_score(y_test, y_pred):.2f}")


# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Определение классификаторов
classifiers = {
    'GradientBoosting': GradientBoostingClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0),
    'AdaBoost': AdaBoostClassifier(),
    'ExtraTrees': ExtraTreesClassifier(),
    'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
    #'LightGBM': LGBMClassifier(),
    'KNeighbors': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'SVM': SVC(kernel='linear'),
}

# Оценка моделей с использованием кросс-валидации
results = {}
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    results[name] = scores.mean()
    print(f'{name}: {scores.mean():.4f}')

# Определение лучшей модели
best_model_name = max(results, key=results.get)
best_model_score = results[best_model_name]
print(f'\nBest model: {best_model_name} with accuracy {best_model_score:.4f}')


# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)

# Обучение модели на обучающем наборе данных
gb_classifier.fit(X_train, y_train)

# Предсказание классов на тестовом наборе данных
y_pred = gb_classifier.predict(X_test)

# Вывод полного отчета
report = classification_report(y_test, y_pred)
print("Классификатор градиентного бустинга")
print(report)

# Классификатор CatBoost на бинарной классификации
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание модели CatBoostClassifier
clf = CatBoostClassifier(iterations=50, depth=10, learning_rate=0.2, loss_function='MultiClass', random_state=42)

# Обучение модели на обучающем наборе данных
clf.fit(X_train, y_train)

# Предсказание классов на тестовом наборе данных
y_pred = clf.predict(X_test)

# Генерация отчета о классификации
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
#Классификатор CatBoost на бинарной классификации
# Вывод отчета
print("Классификатор CatBoost на бинарной классификации")
print(f'Accuracy: {accuracy:.2f}')
print("Classification Report:")
print(report)

# 3. AdaBoost

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание и обучение классификатора AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=1)
ada_classifier = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, learning_rate=0.1, random_state=42)
ada_classifier.fit(X_train, y_train)

# Прогнозирование классов на тестовом наборе данных
y_pred = ada_classifier.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение классификатора QDA
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = clf.predict(X_test)

# Вывод метрик классификации
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)


# lightgbm
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение классификатора LightGBM
clf = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=100)
clf.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = clf.predict(X_test)

# Вывод метрик классификации
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)


# ExtraTreesClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение классификатора Extra Trees
clf = ExtraTreesClassifier(n_estimators=100, max_features='sqrt', random_state=42)
clf.fit(X_train, y_train)

# Прогнозирование и оценка точности
y_pred = clf.predict(X_test)

# Вывод метрик классификации
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("ExtraTreesClassifier")
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
