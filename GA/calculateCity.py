import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

f = open('TSP.csv', 'r')
reader = csv.reader(f)

df = pd.read_csv('TSP.csv', names=['x', 'y'])

cityCoordinate = []
city_x = []
city_y = []
#limit_city = 20

for line in reader:
    line0 = float(line[0])
    line1 = float(line[1])
    city = [line0, line1]
    cityCoordinate.append(city)
    city_x.append(line0)
    city_y.append(line1)
    #if len(cityCoordinate) == limit_city:
    #    break

f.close()

for i in range(0, len(cityCoordinate)):
    print("도시 %d: x: %f y: %f" % (i+1, cityCoordinate[i][0], cityCoordinate[i][1]))

data = df[['x', 'y']]

scaler = MinMaxScaler()
data_scale = scaler.fit_transform(data)

k = 4

# 그룹 수, random_state 설정
model = KMeans(n_clusters=k, random_state=10)

# 정규화된 데이터에 학습
model.fit(data_scale)

# 클러스터링 결과 각 데이터가 몇 번째 그룹에 속하는지 저장
df["cluster"] = model.fit_predict(data_scale)

for i in range(k):
    plt.scatter(df.loc[df['cluster'] == i, 'x'], df.loc[df['cluster'] == i, 'y'],
                label='cluster' + str(i), s=10)
plt.xlabel('X', size=12)
plt.ylabel('Y', size=12)
plt.show()

cities_of_cluster = []
idxs_list = []
for i in range(0, k):
    tmp = [cities for cities in df[df['cluster'] == i].index]
    idxs_list.append(tmp)
    tmp_list = []
    for k in tmp:
        tmp_list.append(tourmanager.getCity(k))
    cities_of_cluster.append(tmp_list)

# plt.scatter(city_x, city_y)
# plt.axis([0, 100, 0, 100])

# # Elbow method
# inertia = []
# for i in range(1, 11):
#     kmeans_plus = KMeans(n_clusters=i, init='k-means++')
#     kmeans_plus.fit(data_scale)
#     inertia.append(kmeans_plus.inertia_)
#
# plt.xlabel('Number of Clusters', size=12)
# plt.ylabel('Inertia', size=12)
# plt.plot(range(1, 11), inertia, marker='o')

# # Silhouette method
# from sklearn.metrics import silhouette_samples
#
# silhouette_vals = []
# for i in range(2, 11):
#     kmeans_plus = KMeans(n_clusters=i, init='k-means++')
#     pred = kmeans_plus.fit_predict(data_scale)
#     silhouette_vals.append(np.mean(silhouette_samples(data_scale, pred, metric='euclidean')))
# plt.plot(range(2, 11), silhouette_vals, marker='o')
# plt.xlabel('Number of Clusters', size=12)
# plt.ylabel('Silhouette', size=12)