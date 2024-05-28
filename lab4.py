
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

# Загрузка и нализ датасета
heart=pd.read_csv("d:\\data_science_24\lab4\heart.csv")
heart.head()
print(heart.head())
print(heart.info())

print(heart.isnull().sum())

minAge=min(heart.age)
maxAge=max(heart.age)
meanAge=heart.age.mean()
print("min =",minAge ,"max =", maxAge , "mean =",meanAge)

X = heart.drop(["target"], axis = 1)
y = heart["target"]

# Классификатор градиентного бустинга

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


# Decision Tree Classifier без подбора параметров
# Импортируем необходимые библиотеки

# Загрузка набора данных
X = heart[['age','trestbps']]
y = heart['target']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(max_depth=1,random_state=42)
dt_classifier.fit(X_train, y_train)

# Предсказание классов на тестовом наборе данных
y_pred = dt_classifier.predict(X_test)

# Оценка производительности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Вывод полного отчета
print("Decision Tree Classifier")
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Визуализация решающей поверхности для первых двух признаков
x_min=heart['age'].min() - 1
x_max=heart['age'].max() + 1
y_min=heart['trestbps'].min() - 1
y_max=heart['trestbps'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
Z = dt_classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(heart['age'],heart['trestbps'], c=y, cmap=plt.cm.bwr, s=20)
plt.xlabel('Age')
plt.ylabel('trestbps')
plt.title('Решающая поверхность с использованием Decision Tree Classifier')
plt.show()










