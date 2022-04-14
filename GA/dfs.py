import random



# 최종 도시들의 정렬이 담긴 클래스
class tour:
    def __init__(self, cities_array, cluster_number):
        self.cities = cities_array
        self.total_length = 0
        self.cities_order = []
        self.cluster_number = cluster_number
        self.cluster_array = []
        self.createClusters(self.cities, self.cluster_number)
        self.bindClusters()

    def getArray(self):
        return self.cities_order

    def createClusters(self, cities, cluster_number):
        cluster_size = len(cities) // cluster_number
        # cluster 분할을 위한 copy_cities를 생성
        copy_cities = cities.copy()

        next_index = 0
        for i in range(cluster_number):

            if i == 0:  # 첫번째 클러스터의 시작은 start_node
                self.cluster_array.append(cluster(copy_cities[next_index]))
                del copy_cities[next_index]
            else:  # 두번째 클러스터부터는 시작값을 랜덤으로 뽑는다.
                next_index = random.randint(0, len(copy_cities) - 1)
                self.cluster_array.append(cluster(copy_cities[next_index]))
                del copy_cities[next_index]

        for i in range(cluster_number):
            # 클러스터의 사이즈 만큼 각 클러스터 시작노드로부터 greedy를 통해 뽑는다
            current_node = self.cluster_array[i].start_node
            for j in range(cluster_size - 1):
                min_distance = 1000000
                nearest_city = current_node
                nearest_city_index = 0
                for k in range(len(copy_cities)):
                    # 현재의 노드로부터 각각의 노드와의 거리비교에서 작은값도출
                    if current_node.getDistance(copy_cities[k]) < min_distance:
                        min_distance = current_node.getDistance(copy_cities[k])
                        nearest_city = copy_cities[k]
                        nearest_city_index = k
                self.cluster_array[i].inputCity(nearest_city)
                del copy_cities[nearest_city_index]
            print(len(self.cluster_array[i].cities_order))
            print(self.cluster_array[i].start_node.index)
            print(self.cluster_array[i].end_node.index)
        print(len(copy_cities))

    def bindClusters(self):
        visited = [-1] * self.cluster_number
        length, array = self.dfsSearch(visited, 0, 0, self.cluster_array)
        temp_array = []
        for i in range(self.cluster_number):
            self.total_length = self.total_length + self.cluster_array[i].length
            temp_array.append(self.cluster_array[array[i]])
            print(array[i])
        self.total_length = self.total_length + length
        self.cluster_array = temp_array.copy()
        self.createCitiesOrder(self.cities, self.cluster_number)
        print(self.total_length)

    # 주어진 cluster들을 dfs기법을 기반으로 연결하여 최적의 코드를 얻는다.
    def dfsSearch(self, visited, depth, total_length, cluster_array):

        if depth == len(cluster_array):
            return total_length, visited

        min_distance = 10000000
        min_array = []

        for node in range(len(cluster_array)):
            if node not in visited:
                print("depth", depth)
                print("node", node)
                if depth == 0 and node != 0:
                    # 시작은 항상 0번째 부터이기에 이후는 필요가 없다.
                    print("check")
                    break
                else:
                    total_length = total_length + cluster_array[visited[depth - 1]].connectClusters(
                        cluster_array[visited[depth]])
                visited[depth] = node
                distance, array = self.dfsSearch(visited, depth + 1, total_length, cluster_array)
                if distance < min_distance:
                    min_distance = distance
                    min_array = array.copy()
                visited[depth] = -1

        return total_length, min_array

    def createCitiesOrder(self, cities, cluster_number):
        for i in range(cluster_number):
            for j in range(len(self.cluster_array[i].cities_order)):
                self.cities_order.append(self.cluster_array[i].cities_order[j])
        self.cities_order.append(cities[0])
        self.total_length = self.total_length + self.cities_order[len(self.cities_order) - 2].getDistance(
            self.cities_order[len(self.cities_order) - 1])


# 각 군집을 담는 클래스
class cluster:
    def __init__(self, start_node):
        self.cities_order = []
        self.start_node = start_node
        self.end_node = start_node
        self.length = 0
        self.cities_order.append(start_node)

    # cluster에 받은 도시를 삽입후 총 길이 갱신(greedy로만으로 cluster를 만들시 사용하기위해서)
    def inputCity(self, city):
        self.cities_order.append(city)
        self.length = self.length + self.cities_order[len(self.cities_order) - 2].getDistance(
            self.cities_order[len(self.cities_order) - 1])
        self.defineEndNode()

    def defineEndNode(self):
        self.end_node = self.cities_order[len(self.cities_order) - 1]

    def connectClusters(self, post_claster):
        return self.end_node.getDistance(post_claster.start_node)




