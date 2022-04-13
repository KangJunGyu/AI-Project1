import math
import random
import csv
import matplotlib.pyplot as plt
import numpy as np

# TSP 도시 생성 클래스
class City:
    # 생성자
    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getIndex(self):
        return self.index

    # 도시 사이 거리 계산
    def getDistance(self, post_city):
        distance = np.linalg.norm(
            np.array([float(self.x), float(self.y)]) - np.array([float(post_city.x), float(post_city.y)]))
        return distance

    # 도시 x, y좌표 리턴
    def __repr__(self):
        return str(self.getX()) + ", " + str(self.getY())


# 여행 매니저
class TourManager:
    # 도착 도시 리스트
    destinationCities = []

    # 도착할 도시(클래스) 추가
    def addCity(self, city):
        self.destinationCities.append(city)

    # 도착한 도시 중에 가져오기
    def getCity(self, index):
        return self.destinationCities[index]

    # 도시 수 가져오기
    def numberOfCities(self):
        return len(self.destinationCities)


# 여행 클래스(적합도 계산)
class Tour:
    def __init__(self, tourmanager, tour=None):
        self.tourmanager = tourmanager
        self.tour = []
        self.fitness = 0.0
        self.distance = 0
        #self.cities_order = []
        #self.total_length = 0

        if tour is not None:
            self.tour = tour
        else:
            for i in range(0, self.tourmanager.numberOfCities()):
                self.tour.append(None)

    # 여행 수 리턴
    def __len__(self):
        return len(self.tour)

    # 특정 여행 리턴
    def __getitem__(self, index):
        return self.tour[index]

    # 특정 여행 설정
    def __setitem__(self, key, value):
        self.tour[key] = value

    # 결과 리턴
    def __repr__(self):
        geneString = 'Start -> '
        for i in range(0, self.tourSize()):
            geneString += str(self.getCity(i)) + ' -> '
        geneString += 'End'
        return geneString

    # 이동할 도시 추가
    def generateIndividual(self):
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            self.setCity(cityIndex, self.tourmanager.getCity(cityIndex))
        random.shuffle(self.tour)
        for cityIndex in range(0, self.tourmanager.numberOfCities()):
            if self.getCity(cityIndex) == self.tourmanager.getCity(0):
                self.tour[0], self.tour[cityIndex] = self.tour[cityIndex], self.tour[0]

    def getCity(self, tourPosition):
        return self.tour[tourPosition]

    def getIndex(self, city):
        return self.tour.index(city)

    def getCount(self):
        count = 0
        for i in range(0, self.tourSize()):
            if self.getCity(i) == None:
                count += 1
        return count

    # TSP에서 이용할 도시 리스트 추가
    def setCity(self, tourPosition, city):
        self.tour[tourPosition] = city
        self.fitness = 0.0
        self.distance = 0

    # fitness(적합도) 계산 후 리턴
    def getFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.getDistance())
        return self.fitness

    # 거리 계산
    def getDistance(self):
        if self.distance == 0:
            tourDistance = 0
            for cityIndex in range(0, self.tourSize()):
                fromCity = self.getCity(cityIndex)
                destinationCity = None
                if cityIndex + 1 < self.tourSize():
                    destinationCity = self.getCity(cityIndex + 1)
                else:
                    destinationCity = self.getCity(0)
                tourDistance += fromCity.getDistance(destinationCity)
            self.distance = tourDistance
        return self.distance

    # 여행 크기
    def tourSize(self):
        return len(self.tour)

    # 특정 도시가 여행에 포함되어있는지 확인
    def containsCity(self, city):
        return city in self.tour


# 인구 클래스
class Population:
    def __init__(self, tourmanager, populationSize, initialise):
        self.tours = []
        self.elit = []
        self.isElit = 0
        for i in range(0, populationSize):
            self.tours.append(None)

        if initialise:
            for i in range(0, populationSize):
                newTour = Tour(tourmanager)
                newTour.generateIndividual()
                self.saveTour(i, newTour)

    # 여행을 리스트에 추가
    def __setitem__(self, key, value):
        self.tours[key] = value

    def __getitem__(self, index):
        return self.tours[index]

    # 여행을 리스트에 추가
    def saveTour(self, index, tour):
        self.tours[index] = tour

    def getTour(self, index):
        return self.tours[index]

    def setElit(self):
        self.elit = self.getFittest()
        self.isElit = 1

    # 가장 적합도가 높은 투어 가져오기
    def getFittest(self):
        fittest = self.tours[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness() <= self.getTour(i).getFitness():
                fittest = self.getTour(i)
        return fittest

    # 여행들 리스트 크기
    def populationSize(self):
        return len(self.tours)

    def getFitnessList(self):
        fitness_list = []
        for i in range(0, self.populationSize()):
            fitness_list.append(self.getTour(i).getFitness())
        return fitness_list