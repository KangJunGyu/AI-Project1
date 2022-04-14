# 1. Random으로 TSP문제 풀기

import csv
from City import City
from Random import tour
import matplotlib.pyplot as plt

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
        cities.append(City(float(row[0]), float(row[1]), i))
        i = i + 1
        plt.scatter(float(row[0]), float(row[1]), c='grey')
        plt.axis([0, 100, 0, 100])

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

with open('solution_random.csv', mode='w', newline='') as sam:
    writer = csv.writer(sam)
    index_array = []
    for row in range(len(parentGene[0].cities_order)):
        index_array.append([parentGene[0].cities_order[row].index])
        print(index_array[row])
        writer.writerow(index_array[row])

print(parentGene[0].total_length)
for j in range(1, len(parentGene[0].cities_order)):
    plt.plot([parentGene[0].cities_order[j].x, parentGene[0].cities_order[j - 1].x], [parentGene[0].cities_order[j].y, parentGene[0].cities_order[j - 1].y], color="blue", linewidth=0.5)

plt.show()