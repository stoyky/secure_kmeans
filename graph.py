from sklearn.cluster import KMeans
import numpy as np
from timeit import Timer
import k_means
import matplotlib.pyplot as plt

MAX_DATA = 50  # max value that an element in the data can go up to.
NUM_FEATURES = 2  # number of attributes in the data set.

secure = [34.989633860000004, 43.529209519999995, 59.48111107999999, 63.79549597, 92.10828853999999, 100.54714739000001, 125.26788816999997, 147.01397721000003, 184.33659162999993, 145.88755629999996]
sk = [0.7052231999999998, 0.7278320000000003, 0.8033452000000003, 0.8634477999999994, 0.8744786000000007, 0.9131806000000005, 0.9584033999999999, 1.0271872999999996, 1.0644048999999995, 1.1130901000000009, 1.1346314]
naive = [21.9553089, 24.0463711, 26.402323199999998, 28.228516600000006, 31.2953153, 33.97615109999998, 37.20303729999998, 39.00175150000001, 44.577136800000005, 47.04522399999996, 49.304454599999985]
x = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000]

# plt.plot(x, secure)
plt.plot(x, naive)
plt.plot(x, sk)
plt.yscale('log')
plt.legend(['secure k-means', 'naive k-means', 'scikit k-means'])
plt.show()

# for i in range(1000, 11000, 1000):
#     print(i)
#     x.append(i)
#     data = np.random.randint(MAX_DATA, size=(i, NUM_FEATURES))
#     t = Timer(lambda: KMeans(n_clusters=3, init='random', n_init=1, random_state=0).fit(data))
#     sk.append(round(t.timeit(number=10), 5)/10)
#     t = Timer(lambda: k_means.kmeans(data, 3, 10, 1))
#     naive.append(round(t.timeit(number=10), 5)/10)
#
# print(secure)
# print(sk)
# print(naive)
# print(x)
