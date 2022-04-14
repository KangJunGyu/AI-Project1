import random
import csv
import matplotlib.pyplot as plt
from City import City
from City import TourManager
from City import Tour
from City import Population
from dfs import tour
from dfs import cluster

# 유전 알고리즘 클래스
class GA:
    def __init__(self, tourmanager, mutationRate=0.1, tournamentSize=30, elitism=True):
        self.tourmanager = tourmanager
        self.mutationRate = mutationRate
        self.tournamentSize = tournamentSize
        self.elitism = elitism

    # 인구 클래스 진화 과정
    def evolvePopulation(self, pop):
        newPopulation = Population(self.tourmanager, pop.populationSize(), False)

        elitismOffset = 0
        if self.elitism:
            newPopulation.saveTour(0, pop.getFittest())
            elitismOffset = 1

        for i in range(elitismOffset, newPopulation.populationSize()):
            parent1 = self.rouletteSelection(pop)
            parent2 = self.rouletteSelection(pop)
            child = self.orderCrossover(parent1, parent2)
            newPopulation.saveTour(i, child)

        for i in range(elitismOffset, newPopulation.populationSize()):
            self.swapMutate(newPopulation.getTour(i))

        return newPopulation

    # 크로스오버 순서교차
    def orderCrossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        startPos = int(random.randint(1, parent1.tourSize()-1))
        endPos = int(random.randint(startPos + 1, parent2.tourSize()))

        for i in range(0, child.tourSize()):
            if i >= startPos and i <= endPos:
                child.setCity(i, parent1.getCity(i))

        for i in range(endPos+1, child.tourSize()):
            for j in range(1, child.tourSize()):
                if not child.containsCity(parent2.getCity(j)):
                    child.setCity(i, parent2.getCity(j))
                    break
        for i in range(0, startPos):
            for j in range(1, child.tourSize()):
                if i == 0:
                    child.setCity(0, parent2.getCity(0))
                    break
                if not child.containsCity(parent2.getCity(j)):
                    child.setCity(i, parent2.getCity(j))
                    break
        #print(child)
        #print(child.getCity(99))
        return child

    def frontOrderCrossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        startPos = int(random.random() * parent1.tourSize())
        endPos = int(random.random() * parent1.tourSize())

        for i in range(0, child.tourSize()):
            if startPos < endPos and i > startPos and i < endPos:
                child.setCity(i, parent1.getCity(i))
            elif startPos > endPos:
                if not (i < startPos and i > endPos):
                    child.setCity(i, parent1.getCity(i))

        for i in range(0, parent2.tourSize()):
            if not child.containsCity(parent2.getCity(i)):
                for ii in range(0, child.tourSize()):
                    if child.getCity(ii) == None:
                        child.setCity(ii, parent2.getCity(i))
                        break
        #print(child)
        return child

    def PMXCrossover(self, parent1, parent2):
        child = Tour(self.tourmanager)

        startPos = random.randint(1, parent1.tourSize() - 2)
        endPos = random.randint(startPos + 1, parent2.tourSize() - 1)

        for i in range(0, child.tourSize()):
            if i >= startPos and i <= endPos:
                child.setCity(i, parent1.getCity(i))

        for i in range(0, child.tourSize()):
            if i < startPos or i > endPos:
                child.setCity(i, parent2.getCity(i))

        duplicate = True
        while duplicate:
            duplicate = False
            target = child[startPos:endPos+1]
            for i in range(0, startPos):
                if child[i] in target:
                    duplicate = True
                    for j in range(startPos, endPos+1):
                        if child.getCity(i) == child.getCity(j):
                            child.setCity(i, parent2.getCity(j))
                            break

            #target = child[:endPos+1]
            for i in range(endPos+1, parent2.tourSize()):
                if child[i] in target:
                    duplicate = True
                    for j in range(startPos, endPos+1):
                        if child.getCity(i) == child.getCity(j):
                            child.setCity(i, parent2.getCity(j))
                            break
            """
            for i in range(1, child.tourSize()):
                if i >= startPos and i <= endPos:
                    # if not child.containsCity(parent2.getCity(i)):
                    if child.getCity(i) in child[startPos:endPos]:
                        duplicate = True
                        for j in range(1, child.tourSize()):
                            if j < startPos or j > endPos:
                                if child.getCity(j) == child.getCity(i):
                                    child.setCity(j, parent2.getCity(i))
            """

        #print(child.tourSize())
        return child

    def cycleCrossover(self, parent1, parent2):
        child = Tour(self.tourmanager)
        ind = 0
        first, second = parent1, parent2

        while True:
            if child.getCount() == 0:
                break
            child[ind] = first[ind]
            ind = first.getIndex(second[ind])
            #print(ind)
            if child[ind] != None and child.getCount() != 0:
                first, second = second, first
                ind = child.getIndex(None)
        return child

    # 변이(도시 위치 서로 바꾸기)
    def swapMutate(self, tour):
        for tourPos1 in range(1, tour.tourSize()):
            if random.random() < self.mutationRate:
                tourPos2 = int(tour.tourSize() * random.uniform(0.05, 1))

                city1 = tour.getCity(tourPos1)
                city2 = tour.getCity(tourPos2)

                tour.setCity(tourPos2, city1)
                tour.setCity(tourPos1, city2)

    def inversionMutate(self, tour):
        startPos = random.randint(1, tour.tourSize() - 2)
        endPos = random.randint(startPos + 1, tour.tourSize() - 1)

        child = Tour(self.tourmanager)
        tmp = 0
        for i in range(startPos, endPos+1):
            for j in range(endPos - tmp, startPos-1, -1):
                child.setCity(i, tour.getCity(j))
                tmp += 1
                break

        for i in range(startPos, endPos+1):
            tour.setCity(i, child.getCity(i))

    # 토너먼트 셀렉션
    def tournamentSelection(self, pop):
        t = 0.6
        n = 3
        tournament = []
        for i in range(0, 2**n):
            tournament.append(random.randint(0, pop.populationSize()-1))

        for i in reversed(range(1, n+1)):
            for j in range(0, 2**(i-1)):
                randomId = random.random()
                if t > randomId:
                    if pop.getTour(2*j).getFitness() > pop.getTour(2*j+1).getFitness():
                        tournament[j] = tournament[2*j]
                    else:
                        tournament[j] = tournament[2*j+1]
                else:
                    if pop.getTour(2*j).getFitness() < pop.getTour(2*j+1).getFitness():
                        tournament[j] = tournament[2*j]
                    else:
                        tournament[j] = tournament[2*j+1]
        return pop.getTour(tournament[0])

        # 좀 이상한 토너먼트(좋음)
        # tournament = Population(self.tourmanager, self.tournamentSize, False)
        # for i in range(0, self.tournamentSize):
        #     randomId = int(random.random() * pop.populationSize())
        #     tournament.saveTour(i, pop.getTour(randomId))
        # fittest = tournament.getFittest()
        # return fittest

    def rankingSelecton(self, pop):
        highest_chrom_idx = self.getSortedFitnessIndex(pop)
        prob_list = [0.5, 0.2, 0.15, 0.1, 0.05]
        p = random.random()
        sum = 0
        for i in range(5):
            sum += prob_list[i]
            if sum >= p:
                idx = i
                break
        return pop.getTour(highest_chrom_idx[idx])

    def getSortedFitnessIndex(self, pop):
        fitness_list = pop.getFitnessList()
        sortedFitness = sorted(fitness_list)
        highest_list = []
        highest_tour_index = []
        for i in range(0, 5):
            highest_fitness = sortedFitness.pop()
            highest_tour_index.append(fitness_list.index(highest_fitness))
        return highest_tour_index

    def elitSelection(self, pop):
        if pop.isElit:
            pop.saveTour(int(random.random() * pop.populationSize()), pop.elit)
        pop.setElit()

        return pop.getTour(int(random.random() * pop.populationSize()))

    def rouletteSelection(self, pop):
        sum = 0
        for i in range(pop.populationSize()):
            sum += pop.getTour(i).getFitness()

        fitnessProb = []
        for i in range(pop.populationSize()):
            fitnessProb.append(pop.getTour(i).getFitness() / sum)

        fitnessSum = 0
        fitness_sumList = []
        for i in fitnessProb:
            fitnessSum += i
            fitness_sumList.append(fitnessSum)
        #print(fitness_sumList)

        rand = random.random()
        #print(rand)
        for i in range(pop.populationSize()):
            if rand <= fitness_sumList[i]:
                #print(pop.getTour(i))
                return pop.getTour(i)

        return None

# 파일 직접 실행시 실행
if __name__ == '__main__':
    f = open('../TSP.csv', 'r')
    reader = csv.reader(f)

    n_cities = 1000
    population_size = 10
    n_generations = 1
    cityCoordinate = []
    city_x = []
    city_y = []
    city_index = []

    #random.seed(100)
    i = 1
    for line in reader:
        line0 = float(line[0])
        line1 = float(line[1])
        city = [line0, line1]
        cityCoordinate.append(city)
        city_x.append(line0)
        city_y.append(line1)
        city_index.append(i)
        i += 1
        if len(cityCoordinate) == n_cities:
            break
    f.close()

    cities = []
    new_cities = []
    sol = []
    parentGene = []

    i = 0
    with open('../TSP.csv', mode='r', newline='') as tsp:
        reader = csv.reader(tsp)
        i = 0
        for row in reader:
            cities.append(City(float(row[0]), float(row[1]), i))
            i = i + 1

    number = int(input("생성할 부모 gene 수를 입력하시오 :  "))
    cluster_number = int(input("생성할 군집의 수를 입력하시오 :  "))

    for i in range(number):
        tournode = tour(cities, cluster_number)
        parentGene.append(tournode)
    print(len(parentGene[0].getArray()))

    # 도시 수 만큼 랜덤 좌표 설정
    tourmanager = TourManager()
    for i in range(n_cities):
        #x = random.randint(200, 800)
        #y = random.randint(200, 800)

        # 도시를 여행 매니저 리스트에 추가
        #tourmanager.addCity(City(x=x, y=y))
        tourmanager.addCity(City(x=city_x[i], y=city_y[i], index=city_index[i]))
        # 각 도시 위치에 점으로 표시
        #cv2.circle(map_original, center=(x, y), radius=10, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
        #plt.scatter(x, y)
        plt.scatter(city_x[i], city_y[i])
        plt.axis([0, 100, 0, 100])


    # map을 이름으로 사진 보여주기
    #cv2.imshow('map', map_original)
    #cv2.waitKey(0)

    tourList = []
    for i in range(0, number):
        tourList.append(parentGene[i].getArray())
    # print(parentGene[0].getArray())
    # print(parentGene[1].getArray())
    randomTour = []
    for i in range(0, number):
        randomTour.append(TourManager())

    for i in range(0, number):
        randomTour[i].setCity(tourList[i])

    randomTourList = []
    for i in range(0, number):
        randomTourList.append(Tour(randomTour[i]))
        randomTourList[i].generate()
    #print(randomTourList[0])

    # Initialize population
    pop = Population(tourmanager, populationSize=number, initialise=False)
    for i in range(0, number):
        pop.saveTour(i, randomTourList[i])
        print(pop.getTour(i))

    print("Initial distance: " + str(pop.getFittest().getDistance()))

    # Evolve population
    ga = GA(tourmanager)

    for i in range(n_generations):
        # population에 대해 유전알고리즘 시행 후 다시 저장
        pop = ga.evolvePopulation(pop)

        # 가장 적합도가 높은 여행
        fittest = pop.getFittest()

        if i == n_generations - 1:
            for j in range(1, n_cities):
                plt.plot([fittest[j].x, fittest[j - 1].x], [fittest[j].y, fittest[j - 1].y], color="blue")
        print(pop.getFittest())
        print("Final distance: " + str(pop.getFittest().getDistance()))
        # 지도에 반영
        #map_result = map_original.copy()

        """
        # 라인 그리기, 적합도가 가장 높은 투어에서 도시 순서대로 라인 그리기
        for j in range(1, n_cities):
            cv2.line(
                map_result,
                pt1=(fittest[j - 1].x, fittest[j - 1].y),
                pt2=(fittest[j].x, fittest[j].y),
                color=(255, 0, 0),
                thickness=3,
                lineType=cv2.LINE_AA
            )

        cv2.putText(map_result, org=(10, 25), text='Generation: %d' % (i + 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7, color=0, thickness=1, lineType=cv2.LINE_AA)
        cv2.putText(map_result, org=(10, 50), text='Distance: %.2fkm' % fittest.getDistance(),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=0, thickness=1, lineType=cv2.LINE_AA)
        cv2.imshow('map', map_result)
        if cv2.waitKey(100) == ord('q'):
            break
        """


    # Print final results
    print("Finished")
    print("Final distance: " + str(pop.getFittest().getDistance()))
    print("Solution:")
    print(pop.getFittest())
    for i in range(0, n_cities):
        print(pop.getFittest().getCity(i).getIndex(), end=' -> ')
    plt.show()

    #cv2.waitKey(0)