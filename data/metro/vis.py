
from math import ceil, floor
import numpy as np
import matplotlib.pyplot as plt

d = np.load('source_data.npy')
max_val = ceil(d.max()/100)
X = [i for i in range(max_val+1)]
Y = [0 for i in range(max_val+1)]
for x in d.flatten():
    if x == 0:
        Y[0] += 1
    else:
        Y[floor(x/100) + 1] += 1

fig = plt.figure()
plt.bar(X,Y,0.4,color="green")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.yscale('log')
plt.title("bar chart")


plt.show()
plt.savefig("barChart.jpg")