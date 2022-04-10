import csv
import matplotlib.pyplot as plt

f = open('TSP.csv', 'r')
reader = csv.reader(f)

cityCoordinate = []
city_x = []
city_y = []
limit_city = 20

for line in reader:
    line0 = float(line[0])
    line1 = float(line[1])
    city = [line0, line1]
    cityCoordinate.append(city)
    city_x.append(line0)
    city_y.append(line1)
    if len(cityCoordinate) == limit_city:
        break

f.close()

for i in range(0, len(cityCoordinate)):
    print("도시 %d: x: %f y: %f" % (i+1, cityCoordinate[i][0], cityCoordinate[i][1]))

plt.scatter(city_x, city_y)
plt.axis([0, 100, 0, 100])
plt.show()

