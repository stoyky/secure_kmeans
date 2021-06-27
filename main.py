from secure_kmeans import *

if __name__ == '__main__':
    k = 3
    epsilon = 1
    max_iter = 15
    # n_samples = 100

    centroids = random_centroids(k)

    timings_sklearn = []
    for i in range(100, 300, 100):
        print(i)
        data = gen_data(k, n_samples=i)
        t = Timer(lambda: KMeans(n_clusters=k, max_iter=max_iter).fit(data))
        timings_sklearn.append(t.timeit(number=1))
    print(timings_sklearn)

    timings_naive = []
    for i in range(100, 300, 100):
        print(i)
        data = gen_data(k, n_samples=i)
        t = Timer(lambda: naive_kmeans(data, centroids, k, epsilon, max_iter, False))
        timings_naive.append(t.timeit(number=1))
    print(timings_naive)

    timings_secure = []
    for i in range(100, 300, 100):
        print(i)
        data = gen_data(k, n_samples=i)
        t = Timer(lambda: secure_kmeans(data, centroids, round, k, epsilon, max_iter, False))
        timings_secure.append(t.timeit(number=1))
    print(timings_secure)