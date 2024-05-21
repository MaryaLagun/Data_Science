
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import oracledb
import seaborn as sns

# Датасет смертность от сердечно-сосудистых заболеваний, рака, диабета, хронических респираторных заболеваний (на 100000 человек населения)
X=pd.read_csv("d:\\data_science_24\lab3\data3.csv")
print(X.describe())

# используется БД Oracle
connection = oracledb.connect(user="PSO_KDF", password="test",
                              host="test", port=1521, service_name="test")
cursor = connection.cursor()

# создание таблицы
cursor.execute("CREATE TABLE PSO_KDF.disease (year VARCHAR2(4),disease VARCHAR2(250),gender VARCHAR2(20),age VARCHAR2(40),region VARCHAR2(40),value NUMBER)")
cursor.execute("INSERT INTO PSO_KDF.disease (year, disease, gender, age, region, value) VALUES (%s, %s, %s, %s, %s, %s)",X)
connection.commit()


query="""
select t1.YEAR, t1.VALUE_HEAD, t2.VALUE_DIAB, t3.VALUE_NEOP, t4.VALUE_CHRONIC from (
select year, value value_head
 from DISEASE t
where t.DISEASE='Смертность от болезней системы кровообращения' and t.GENDER='Итого' and t.AGE='Итого' and t.REGION='Итого') t1,
(select year, value value_diab
 from DISEASE t
where t.DISEASE='Смертность от сахарного диабета' and t.GENDER='Итого' and t.AGE='Итого' and t.REGION='Итого') t2,
(select year, value value_neop
 from DISEASE t
where t.DISEASE='Смертность от злокачественных новообразований' and t.GENDER='Итого' and t.AGE='Итого' and t.REGION='Итого') t3,
(select year, value value_chronic
from DISEASE t
where t.DISEASE='Смертность от хронических респираторных заболеваний' and t.GENDER='Итого' and t.AGE='Итого' and t.REGION='Итого') t4
where t1.YEAR = t2.YEAR and t2.YEAR = t3.YEAR and t3.YEAR = t4.YEAR
order by 1
  """
cursor.execute(query)

users = cursor.fetchall()
for user in users:
    print(user)

X1 = []
Y = []
X2=[]
X3=[]
X4=[]
for ROWS in users:
    X1.append(float(ROWS[1]))
    X2.append(float(ROWS[2]))
    X3.append(float(ROWS[3]))
    X4.append(float(ROWS[4]))
    Y.append(float(ROWS[0]))

plt.scatter(Y,X1,color='black',alpha=0.75)
plt.title('Смертность от болезней системы кровообращения по годам')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.scatter(Y,X2,color='blue',alpha=0.75)
plt.title('Смертность от сахарного диабета по годам')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.scatter(Y,X3,color='green',alpha=0.75)
plt.title('Смертность от злокачественных новообразований по годам')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.scatter(Y,X4,color='red',alpha=0.75)
plt.title('Смертность от хронических респираторных заболеваний по годам')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

query="""
select to_number(substr(t.age,1,2)) age, sum(t.VALUE) as value_all
from DISEASE t
where t.DISEASE='Смертность от болезней системы кровообращения' and t.GENDER='Итого' and t.AGE<>'Итого' and t.REGION='Итого'
group by t.AGE
order by 1
"""
cursor.execute(query)
users2 = cursor.fetchall()
for user2 in users2:
    print(user2)

X_age = []
Y_age = []

for ROWS in users2:
    X_age.append(float(ROWS[0]))
    Y_age.append(float(ROWS[1]))

plt.plot(X_age, Y_age)
plt.title('Зависимость смертности от возраста')
plt.xlabel('Возраст')
plt.ylabel('Количество случаев')
plt.show()

connection.commit()
connection.close()




