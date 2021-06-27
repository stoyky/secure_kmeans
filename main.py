from sklearn.cluster import KMeans
from secure_kmeans import *
from timeit import Timer


def graph_performance(sk, naive, secure, range_start, range_end, step):
    x = [i for i in range(range_start, range_end, step)]

    plt.plot(x, sk)
    plt.plot(x, naive)
    plt.plot(x, secure)

    plt.yscale('log')
    plt.legend(['scikit k-means', 'naive k-means', 'secure k-means'])
    plt.show()

if __name__ == '__main__':
    k = 3
    epsilon = 1
    max_iter = 15

    range_start = 1000
    range_end = 3000
    step = 1000
    n_timings = 10

    centroids = random_centroids(k)

    timings_sklearn = []
    for i in range(range_start, range_end, step):
        print(i)
        data = gen_data(k, n_samples=i)
        t = Timer(lambda: KMeans(n_clusters=k, max_iter=max_iter).fit(data))
        timings_sklearn.append(t.timeit(number=n_timings))
    print(timings_sklearn)

    timings_naive = []
    for i in range(range_start, range_end, step):
        print(i)
        data = gen_data(k, n_samples=i)
        t = Timer(lambda: naive_kmeans(data, centroids, k, epsilon, max_iter, False))
        timings_naive.append(t.timeit(number=n_timings))
    print(timings_naive)

    timings_secure = []
    for i in range(range_start, range_end, step):
        print(i)
        data = gen_data(k, n_samples=i)
        t = Timer(lambda: secure_kmeans(data, centroids, k, epsilon, max_iter, False))
        timings_secure.append(t.timeit(number=n_timings))
    print(timings_secure)

    graph_performance(timings_sklearn, timings_naive, timings_secure, range_start, range_end, step)