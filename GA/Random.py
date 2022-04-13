import numpy as np
import csv
import random
from City import City
from City import TourManager
from City import Tour
from City import Population
        
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
        start_city = copy_cities[0]
        self.cities_order.append(copy_cities[next_index]) #시작 도시를 먼저 삽입후 시작
        del copy_cities[next_index]
        
        for i in range(len(copy_cities)):
            next_index = random.randint(0,len(copy_cities)-1)#copy_cities에서 랜덤 인덱스값 받아오기
            #print(next_index)
            self.cities_order.append(copy_cities[next_index]) 
            del copy_cities[next_index]
            
            pre_city = self.cities_order[i]
            post_city = self.cities_order[i+1]
            self.total_length = self.total_length + pre_city.getDistance(post_city)
        
        #1000개의 배열이 끝난뒤 TSP이기 때문에 마지막으로 시작도시를 넣어준다.       
        self.cities_order.append(start_city)
        self.total_length = self.total_length + pre_city.getDistance(start_city)

cities = []
new_cities = []
sol = []
parentGene = []
tourmanager = TourManager()

i = 0
#csv값들을 받아 도시 클래스 배열에 대입
with open('../TSP.csv', mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    i = 0
    for row in reader:
        tourmanager.addCity(City(row[0], row[1], i))
        i = i + 1
        '''
        ***객체배열이 잘 만들어졌는지 확인하는 코드***
        print(cities[i].x , cities[i].y)
        i = i +1
        '''
        
number = int(input("생성할 부모 gene 수를 입력하시오 :  "))
for i in range(number):
    tournode = tour(tourmanager.getCity(i))
    tournode.makeOrder() 
    parentGene.append(tournode)
    
for i in range(number):
    k = parentGene[i].total_length
    last = parentGene[i].cities_order[len(parentGene[i].cities_order)-1].index
    print(len(parentGene[0].cities_order))
    print(last)

with open('sample.csv', mode = 'w', newline='') as sam:
    writer = csv.writer(sam)
    index_array = []
    for row in range(len(parentGene[0].cities_order)):
        index_array.append([parentGene[0].cities_order[row].index])
        print(index_array[row])
        writer.writerow(index_array[row])