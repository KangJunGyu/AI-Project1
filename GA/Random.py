import numpy as np
import csv
import random
from City import City

class tour:
    def __init__(self, cities_array):
        self.cities = cities_array
        self.total_length = 0
        self.cities_order = []

    def getArray(self):
        return self.cities_order

    # cities 배열 생성
    def makeOrder(self, cities):
        copy_cities = cities.copy();
        # valid_index = len(cities)
        next_index = 0
        start_city = copy_cities[0]
        self.cities_order.append(copy_cities[next_index])  # 시작 도시를 먼저 삽입후 시작
        del copy_cities[next_index]

        for i in range(len(copy_cities)):
            next_index = random.randint(0, len(copy_cities) - 1)  # copy_cities에서 랜덤 인덱스값 받아오기
            # print(next_index)
            self.cities_order.append(copy_cities[next_index])
            del copy_cities[next_index]

            pre_city = self.cities_order[i]
            post_city = self.cities_order[i + 1]
            self.total_length = self.total_length + pre_city.getDistance(post_city)

        # 1000개의 배열이 끝난뒤 TSP이기 때문에 마지막으로 시작도시를 넣어준다.
        self.cities_order.append(start_city)
        self.total_length = self.total_length + pre_city.getDistance(start_city)



