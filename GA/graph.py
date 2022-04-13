import numpy as np
import matplotlib.pyplot as plt

"""
x = np.arange(0, 31)
y = [0, 0.145, 0.187, 0.214, 0.235, 0.217,
     0.243, 0.251, 0.364, 0.489, 0.614,
     0.698, 0.764, 0.735, 0.682, 0.625,
     0.519, 0.492, 0.482, 0.421, 0.315,
     0.402, 0.489, 0.543, 0.612, 0.582,
     0.576, 0.532, 0.451, 0.336, 0.268]              # [ 0  1  4  9 16]
"""

x = [0, 1]
y = [1, 0]
plt.plot(x, y)
plt.xlabel('Time Ratio')
plt.ylabel('Weight')
plt.xlim([0, 1])
plt.ylim([0, 1])

plt.show()