from sklearn.cluster import KMeans
from secure_kmeans import *
from timeit import Timer
import cProfile


def graph_performance(sk, naive, secure, range_start, range_end, step):
    """
    Utility function to plot time as function of how many data points
    are to be clustered
    """
    x = [i for i in range(range_start, range_end, step)]

    plt.plot(x, sk)
    plt.plot(x, naive)
    plt.plot(x, secure)

    plt.yscale('log')
    plt.legend(['scikit k-means', 'naive k-means', 'secure k-means'])
    plt.show()


def graph_calls():
    """
    Utility function to create a callgraph for visualization in
    gprof2dot or snakeviz
    To generate call graph .png run:
    gprof2dot -f pstats performance.prof | dot -Tpng -o output.png
    """

    data = gen_data(k, n_samples=1000)

    with cProfile.Profile() as pr:
        pr.run("secure_kmeans(data, centroids, k, epsilon, max_iter, False)")

    pr.dump_stats("performance.prof")


if __name__ == '__main__':
    k = 3
    epsilon = 1
    max_iter = 15

    range_start = 1000
    range_end = 20000
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

    print(timings_sklearn)
    print(timings_naive)
    print(timings_secure)
    graph_performance(timings_sklearn, timings_naive, timings_secure, range_start, range_end, step)
