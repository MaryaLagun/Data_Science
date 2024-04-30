#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[13]:


unique_numbers = {110, 105, 103, 104, 105}
unique_numbers.add(99)
print("unique_numbers:", unique_numbers)


# In[ ]:





# In[16]:


# Сложность алгоритма o(n)
def input_grade():
   try:
       # Попытка преобразовать ввод в число
       unique_numbers = {110, 102, 103, 55, 454}
       grade = float(input("Введите ваше число: "))
       unique_numbers.add(grade)
   except ValueError:
       # Обработка исключения, если ввод не является числом
       print("Ошибочный ввод! Пожалуйста, введите число.")
   finally:
       # Этот блок кода выполнится независимо от того, возникло исключение или нет
       print("unique_numbers:", unique_numbers)

# Вызов функции для ввода оценки
input_grade()



# In[ ]:




