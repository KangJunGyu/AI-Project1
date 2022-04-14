import numpy as np
import csv
import random
from City import City
from Random import tour

cities = []
new_cities = []
sol = []
parentGene = []

i = 0
# csv값들을 받아 도시 클래스 배열에 대입
with open('../TSP.csv', mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    i = 0
    for row in reader:
        cities.append(City(row[0], row[1], i))
        i = i + 1
        '''
        ***객체배열이 잘 만들어졌는지 확인하는 코드***
        print(cities[i].x , cities[i].y)
        i = i +1
        '''

number = int(input("생성할 부모 gene 수를 입력하시오 :  "))
for i in range(number):
    tournode = tour(cities)
    tournode.makeOrder(cities)
    parentGene.append(tournode)

for i in range(number):
    k = parentGene[i].total_length
    last = parentGene[i].cities_order[len(parentGene[i].cities_order) - 1].index
    print(len(parentGene[0].cities_order))
    print(last)

with open('sample.csv', mode='w', newline='') as sam:
    writer = csv.writer(sam)
    index_array = []
    for row in range(len(parentGene[0].cities_order)):
        index_array.append([parentGene[0].cities_order[row].index])
        print(index_array[row])
        print(parentGene[0].total_length)
        writer.writerow(index_array[row])