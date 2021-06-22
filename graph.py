from sklearn.cluster import KMeans
import numpy as np
from timeit import Timer
import k_means
import matplotlib.pyplot as plt

MAX_DATA = 50  # max value that an element in the data can go up to.
NUM_FEATURES = 2  # number of attributes in the data set.

secure = [34.989633860000004, 43.529209519999995, 59.48111107999999, 63.79549597, 92.10828853999999, 100.54714739000001, 125.26788816999997, 147.01397721000003, 184.33659162999993, 145.88755629999996]
sk = [0.003921, 0.011429, 0.012303999999999999, 0.008525000000000001, 0.008748, 0.016375, 0.01094, 0.013822000000000001, 0.015851, 0.015127999999999999]
naive = [0.23128500000000002, 0.487852, 0.714677, 0.959487, 1.1666029999999998, 1.423988, 1.6614149999999999, 1.8916239999999998, 2.130425, 2.365872]
x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

plt.plot(x, secure)
plt.plot(x, naive)
plt.plot(x, sk)
plt.yscale('log')
plt.legend(['secure k-means', 'naive k-means', 'scikit k-means'])
plt.show()

for i in range(1000, 11000, 1000):
    print(i)
    x.append(i)
    data = np.random.randint(MAX_DATA, size=(i, NUM_FEATURES))
    t = Timer(lambda: KMeans(n_clusters=3, init='random', n_init=1, random_state=0).fit(data))
    sk.append(round(t.timeit(number=10), 5)/10)
    t = Timer(lambda: k_means.kmeans(data, 3, 10, 1))
    naive.append(round(t.timeit(number=10), 5)/10)

print(secure)
print(sk)
print(naive)
print(x)
