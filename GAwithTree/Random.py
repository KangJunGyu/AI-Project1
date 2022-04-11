# -*- coding: utf-8 -*-

import numpy as np
import csv
import random

#도시 클래스
class city:
    def __init__(self , x , y, index):
        self.x = x
        self.y = y
        self.index = index
    
    def getDistance(self, post_city):
        distance = np.linalg.norm(np.array([float(self.x), float(self.y)]) - np.array([float(post_city.x), float(post_city.y)]))
        return distance
        
class tour:
    def __init__(self, cities_array):
        self.cities = cities_array
        self.total_length = 0
        self.cities_order = []
        
    #cities 배열 생성
    def makeOrder(self):
        copy_cities = cities.copy();
        #valid_index = len(cities)
        next_index = 0
        self.cities_order.append(copy_cities[next_index]) #시작 도시를 먼저 삽입후 시작
        del copy_cities[next_index]

        print(len(copy_cities))
        for i in range(len(copy_cities)):
            next_index = random.randint(0,len(copy_cities)-1)#copy_cities에서 랜덤 인덱스값 받아오기
            #print(next_index)
            self.cities_order.append(copy_cities[next_index]) 
            del copy_cities[next_index]
            
            pre_city = self.cities_order[i]

            post_city = self.cities_order[i+1]
            print(pre_city.getDistance(post_city))
            self.total_length = self.total_length + pre_city.getDistance(post_city)
        return self.total_length
        
cities = []
new_cities = []
sol = []
parentGene = []


i=0
#csv값들을 받아 도시 클래스 배열에 대입
with open('TSP.csv', mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    i = 0
    for row in reader:
        cities.append(city(row[0], row[1], i))
        i = i + 1
        '''
        ***객체배열이 잘 만들어졌는지 확인하는 코드***
        print(cities[i].x , cities[i].y)
        i = i +1
        '''

tournode = []
number = int(input("생성할 부모 gene 수를 입력하시오 :  "))
for i in range(number):
    tournode.append(tour(cities))
    print(tournode[i].makeOrder())
    parentGene.append(tournode[i])
    
for i in range(number):
    k = parentGene[i].total_length
    print(k)



