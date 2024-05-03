#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Лаба2 Вариант 6
class Car:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year

    def start_engine(self):
        return "Двигатель запущен!"

class ElectricCar(Car):
    def __init__(self, brand, year, battery_size):
        Car.__init__(self, brand, year)
        self.battery_size = battery_size

# Создание экземпляра класса
my_car = Car("Geely", 2024)
my_car_el=ElectricCar("BMW", 2024, 220)
print(my_car.start_engine()) 
print(my_car_el.start_engine()) 


# In[2]:


# Лаба2 Вариант 7
class Car:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year

    def start_engine(self):
        return "Двигатель запущен!"

class ElectricCar(Car):
    def __init__(self, brand, year, battery_size):
        Car.__init__(self, brand, year)
        self.battery_size = battery_size

    def start_engine(self):
        return "Тихий запуск двигателя!"


# Создание экземпляра класса
my_car = Car("Geely", 2024)
my_car_el=ElectricCar("BMW", 2024, 220)

# Использование атрибутов и методов объекта
print(my_car.start_engine()) 
print(my_car_el.start_engine()) 


# In[6]:


# Лаба2 Вариант 8
class Car:
    def __init__(self, brand, year):
        self.brand = brand
        self.year = year

    def start_engine(self):
        return "Двигатель запущен!"

class ElectricCar(Car):
    def __init__(self, brand, year, battery_size):
        super().__init__(brand, year)
        self.battery_size = battery_size

    def start_engine(self):
        return "Тихий запуск двигателя!"


# Создание экземпляра класса
my_car = Car("Geely", 2024)
my_car_el=ElectricCar("BMW", 2024, 220)

# Использование атрибутов и методов объекта
print(my_car.start_engine()) 
print(my_car_el.start_engine()) 


# In[ ]:




