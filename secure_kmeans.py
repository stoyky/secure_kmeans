import matplotlib.pyplot as plt
from phe import paillier
from random_share import naive_generate_share, naive_reconstruct_share
from ssp import ssp
from yaos import *
from calculate_terms import *
from sklearn.datasets import make_blobs

MAX_DATA = 50  # max value that an element in the data can go up to.
NUM_FEATURES = 2  # number of attributes in the data set.

p = 347
q = 349
n = p * q

pubkey = paillier.PaillierPublicKey(n)
prikey = paillier.PaillierPrivateKey(pubkey, p, q)


def create_random_data(data):
    """
    Allocates owner randomly to data.
    :param data: data to randomly allocate.
    :return: p0's and p1's data ownership.
    """
    p0_data = []
    p1_data = []

    for column_idx in range(data.shape[1]):
        random_value = random.randint(0, data.shape[0])  # choose random value between 0 - 100.

        # randomly select indices for A of random value amount.
        possible_index = list(range(data.shape[0]))
        p0_idx = []  # index of p0 owned elements of column X.

        for i in range(random_value):
            random_index = random.choice(possible_index)
            possible_index.remove(random_index)
            p0_idx.append(random_index)

        p0_data.append(sorted(p0_idx))
        p1_idx = list(set(range(data.shape[0])) - set(p0_idx))
        p1_data.append(sorted(p1_idx))

    return p0_data, p1_data


def idx_owner(idx, p0_data):
    """
    Gets the owner of of data point.
    :param idx: index of data point to check.
    :param p0_data: list of all p0's owned data points.
    :return: owner of each feature of data.
    """
    feature1_owner_p0 = False
    feature2_owner_p0 = False

    if idx in p0_data[0]:
        feature1_owner_p0 = True

    if idx in p0_data[1]:
        feature2_owner_p0 = True

    return feature1_owner_p0, feature2_owner_p0


def random_centroids(k):
    """
    Generates random centroids.
    :param k: number of centroids to generate.
    :return: randomly generted k centroids.
    """
    centroids = []
    for i in range(k):
        centroids.append([np.random.randint(0, MAX_DATA), np.random.randint(0, MAX_DATA)])

    return centroids


def dist_euclid(x1, y1, x2, y2):
    """
    Performs euclidean distance.
    :return: euclidean distance between two coordinates.
    """
    return np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2) % n)


def secure_kmeans(data, centroids, k, epsilon, max_iter, plot=False):
    """
    Perform secure kmeans as defined by Jagannathan and Wright.
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.387.826&rep=rep1&type=pdf

    We refer to the involved parties as p0, p1, ... , pn

    :param data: data to perform secure kmeans on.
    :param centroids: centroids to use.
    :param k: number of centroids.
    :param epsilon:
    :param max_iter:
    :param plot:
    :return:
    """

    p0_data, p1_data = create_random_data(data)
    converged = False
    current_iter = 0

    while not converged:
        print("Current iteration: {0}".format(current_iter))
        if current_iter < max_iter:
            current_iter += 1
            # calculating the distance between point and centroid.
            closest_cluster = []  # keeps track of the closest centroid for each data point (index mapping)

            # SECURE K-MEANS ALGORITHM
            # 1. calculate the closest centroid.
            for idx in range(data.shape[0]):  # iterates 100 times.
                p0_owns_feature1, p0_owns_feature2 = idx_owner(idx, p0_data)
                p0_shares = []  # shares after computing the SSP (calculating the 6 terms.)
                p1_shares = []

                p0_old_centroids = []
                p1_old_centroids = []
                for x, y in centroids:
                    p0_x, p1_x = naive_generate_share(x, n)
                    p0_y, p1_y = naive_generate_share(y, n)

                    p0_old_centroids.append((p0_x, p0_y))
                    p1_old_centroids.append((p1_x, p1_y))

                    p0_term1 = calculate_p0_term1(data[idx], p0_owns_feature1, p0_owns_feature2, n)
                    p1_term1 = calculate_p1_term1(data[idx], p0_owns_feature1, p0_owns_feature2, n)

                    p0_term2 = calculate_term2([p0_x, p0_y], n)  # calculate the summation of her centroid shares.
                    p1_term3 = calculate_term3([p1_x, p1_y], n)  # calculate the summation

                    p0_term4, p1_term4 = ssp(pubkey, prikey, n, [p0_x, p0_y], [p1_x, p1_y], mult=2)
                    p0_term5, p1_term5 = ssp(pubkey, prikey, n, [p0_x, p0_y], data[idx].tolist(), mult=2)
                    p0_term6, p1_term6 = ssp(pubkey, prikey, n, data[idx].tolist(), [p1_x, p1_y], mult=2)

                    p0_complete_term = (p0_term1 + p0_term2 + p0_term4 - p0_term5 - p0_term6) % n
                    p1_complete_term = (p1_term1 + p1_term3 + p1_term4 - p1_term5 - p1_term6) % n

                    p0_shares.append(p0_complete_term)
                    p1_shares.append(p1_complete_term)

                p_bits_cc, input_p_bits_cc = generate_p_bits_cc()  # Get the p_bits for the circuits and inputs for closest cluster
                closest = closestcluster(p0_shares, p1_shares, p_bits_cc,
                                         input_p_bits_cc,
                                         n)  # Returns the index of the smallest sum, i.e. closest cluster
                closest_cluster.append(closest)  # add the closest cluster for the data point.

            # 2. recompute the mean
            for _k in range(k):
                data_point_indicies = np.where(np.asarray(closest_cluster) == _k)[0]

                p0_sum = [0, 0]  # track of p0's summation for current cluster centroids for data point she owns. (
                # (feature1, feature2)
                p0_owns_count = [0, 0]  # track of number of data points she owns in current cluster centroid.
                p1_sum = [0, 0]
                p1_owns_count = [0, 0]

                for data_point_index in data_point_indicies:
                    p0_owns_feature1, p0_owns_feature2 = idx_owner(data_point_index, p0_data)

                    if p0_owns_feature1:
                        p0_sum[0] += data[data_point_index][0]
                    else:
                        p1_sum[0] += data[data_point_index][0]

                    if p0_owns_feature2:
                        p0_sum[1] += data[data_point_index][1]
                    else:
                        p1_sum[1] += data[data_point_index][1]

                    if p0_owns_feature1 and p0_owns_feature2:  # p0 owns everything. # 1 1
                        p0_owns_count = [count + 1 for count in p0_owns_count]
                    elif (not p0_owns_feature1) and (not p0_owns_feature2):  # p1 owns everything # 0 0
                        p1_owns_count = [count + 1 for count in p1_owns_count]
                    elif p0_owns_feature1:  # 1 0
                        p0_owns_count[0] += 1
                        p1_owns_count[1] += 1
                    else:  # 0 1
                        p0_owns_count[1] += 1
                        p1_owns_count[0] += 1

                p_bits_rm, input_p_bits_rm = generate_p_bits_rm()

                for j in range(NUM_FEATURES):
                    centroids[_k][j] = recomputemean(p0_sum[j], p1_sum[j], p0_owns_count[j], p1_owns_count[j],
                                              p_bits_rm, input_p_bits_rm)

            # 3. check for termination.
            p0_centroid_shares = []
            p1_centroid_shares = []
            for centroid, p0_old_centroid, p1_old_centroid in zip(centroids, p0_old_centroids, p1_old_centroids):
                p0_x, p1_x = naive_generate_share(centroid[0], n)
                p0_y, p1_y = naive_generate_share(centroid[1], n)

                # calculate term 1 = old centroids squared.
                # expand brackets = (ua1 ^ 2 + ub1 ^ 2 + 2ua1b1) + (ua2 ^ 2 + ub2 ^ 2 + 2ua2b2)
                p0_old_squared = 0
                p1_old_squared = 0

                for i in range(NUM_FEATURES):  # iterate over all features.
                    p0_old_squared += (p0_old_centroid[i] ** 2) % n
                    p1_old_squared += (p1_old_centroid[i] ** 2) % n
                p0_share_term1, p1_share_term1 = ssp(pubkey, prikey, n, list(p0_old_centroid), list(p1_old_centroid), mult=2)  # calculate 2*uab

                # calculate term 2 = new centroids squared.
                p0_new_squared = 0
                p1_new_squared = 0

                p0_new_centroid = (p0_x, p0_y)
                p1_new_centroid = (p1_x, p1_y)

                for i in range(NUM_FEATURES):  # iterate over all features.
                    p0_new_squared += (p0_new_centroid[i] ** 2) % n
                    p1_new_squared += (p1_new_centroid[i] ** 2) % n
                p0_share_term2, p1_share_term2 = ssp(pubkey, prikey, n, list(p0_new_centroid), list(p1_new_centroid),
                                                   mult=2)  # calculate 2*uab

                # calculate term 3 = (new_a + new_b) (old_a + old_b) +  (new_a + new_b) (old_a + old_b)
                p0 = 0  # new_a * old_a
                p1_old_new = 0  # new_b * old_b

                for i in range(NUM_FEATURES):
                    p0 += (2 * (p0_new_centroid[i] * p0_old_centroid[i])) % n
                    p1_old_new += (2 * (p1_new_centroid[i] * p1_old_centroid[i])) % n

                # new_b * old_a
                p0_share_term3, p1_share_term3 = ssp(pubkey, prikey, n,  list(p1_new_centroid), list(p0_old_centroid),
                                                   mult=2)  # calculate 2*uab

                # new_a * old_b
                p0_share_term4, p1_share_term4 = ssp(pubkey, prikey, n, list(p0_new_centroid), list(p1_old_centroid),
                                                   mult=2)  # calculate 2*uab
                p0_centroid_term = (p0_old_squared + p0_share_term1 + p0_new_squared +\
                                      p0_share_term2 - p0 - p0_share_term3 - p0_share_term4) % n
                p1_centroid_term = (p1_old_squared + p1_share_term1 + p1_new_squared +\
                                      p1_share_term2 - p1_old_new - p1_share_term3 - p1_share_term4) % n

                p0_centroid_shares.append(p0_centroid_term)
                p1_centroid_shares.append(p1_centroid_term)

            p_bits_tm = np.random.randint(0, MAX_DATA, (3, 2))  # 3 values; findbiggest(3)
            input_p_bits_tm = np.random.randint(0, MAX_DATA, (4, 2))  # 4 for findbiggest
            below_epsilon = 1

            for p0_share, p1_share in zip(p0_centroid_shares, p1_centroid_shares):
                ab = naive_reconstruct_share(p0_share, p1_share, n)  # epsilon has to be a integer. 0.2 * 10 = 2.
                # returns 0 if any is above epsilon, thus 1*0 = 0. if all is below epsilon, remains 1.
                below_epsilon *= terminate(ab, epsilon, p_bits_tm, input_p_bits_tm)

            converged = bool(below_epsilon)

            # # 4. plot some nice graphs.
            if plot:
                plot_and_save(data, k, centroids, closest_cluster, current_iter, secure=True)

        else:
            print("MAX ITERATIONS REACHED.")
            converged = True  # too many iterations.

    # 5. Allocate cluster centers to p0 and p1
    p0_cluster_centers = []
    p1_cluster_centers = []
    for _k in range(k):
        data_point_indices = np.where(np.asarray(closest_cluster) == _k)[0]

        for idx in data_point_indices:
            p0_owns_f1, p0_owns_f2 = idx_owner(idx, p0_data)
            if p0_owns_feature1 and p0_owns_feature2:  # p0 owns all features
                p0_cluster_centers.append(tuple(centroids[_k]))
                # p1 owns all features.
            elif not(p0_owns_f1) and not(p0_owns_f2):
                p1_cluster_centers.append(tuple(centroids[_k]))
                # they share features
            else:
                p0_cluster_centers.append(tuple(centroids[_k]))
                p1_cluster_centers.append(tuple(centroids[_k]))

    # PRINT IF p0 AND p1 WANTS TO KNOW THEIR CLUSTER CENTERS.
    # p0_cluster_centers = set(p0_cluster_centers)
    # p1_cluster_centers = set(p1_cluster_centers)
    #
    # print("p0's cluster centers are: {0}".format(p0_cluster_centers))
    # print("p1's cluster centers are: {0}".format(p1_cluster_centers))
    print("done")


def naive_kmeans(data, centroids, k, epsilon, max_iter, plot=False):
    """
    Perform naive kmeans.
    :param data: data to apply kmeans.
    :param centroids: centroids to use.
    :param k: number of centroids.
    :param epsilon: threshold for centroids convergence.
    :param max_iter: threshold for maximum number of iterations.
    :param plot: plot graph if true.
    """
    converged = False
    current_iter = 0

    while not converged:
        print("Current iteration: {0}".format(current_iter))
        if current_iter < max_iter:
            current_iter += 1
            # calculating the distance between point and centroid.
            closest_cluster = []  # keeps track of the closest centroid for each data point (index mapping)
            # NAIVE K-MEANS
            # 1. calculate the closest centroid
            for point in data:
                temp_dist = []
                for centroid in centroids:
                    temp_dist.append(dist_euclid(point[0], point[1], centroid[0], centroid[1]))
                closest_cluster.append(np.argmin(temp_dist))

            # 2. calculate the new centroid
            centroids_avg = []
            for i in range(k):
                summ = 0
                summ1 = 0
                gh = np.where(np.asarray(closest_cluster) == i)[0]

                for g in gh:
                    summ += data[g][0]
                    summ1 += data[g][1]

                if len(gh) > 0:
                    summ = summ / len(gh)
                    summ1 = summ1 / len(gh)

                centroids_avg.append((summ, summ1))

            # 3. check for termination
            naive_converg = 1
            for i in range(k):
                u = dist_euclid(centroids[i][0], centroids[i][1], centroids_avg[i][0], centroids_avg[i][1])
                if u <= epsilon:
                    naive_converg *= 1
                else:
                    naive_converg *= 0

            # 4. Plot nice images
            if plot:
                plot_and_save(data, k, centroids, closest_cluster, current_iter, secure=False)

            if naive_converg == 1:
                print("naive k-means terminated.")
                return
            else:
                centroids = centroids_avg
        else:
            return


def plot_and_save(data, k,  centroids, closest_cluster, current_iter, secure=False):
    """
    Plots graphs and saves to file location.
    :param data: data to plot.
    :param k: number of centroids.
    :param centroids: centroids.
    :param closest_cluster: array stating which data belongs to which cluster.
    :param current_iter:
    :param secure:
    :return:
    """
    colours = ["red", "blue", "green", "purple", "yellow"]
    plt.Figure()
    for i in range(k):
        plt.scatter(centroids[i][0], centroids[i][1], color="black", sizes=[100.0], marker='X', zorder=3)
        data_point_indices = np.where(np.asarray(closest_cluster) == i)[0]

        for index in data_point_indices:
            plt.scatter(data[index][0], data[index][1], color=colours[i])

    if secure:
        save_name = r"images/secure_kmeans_{0}.png".format(current_iter)
    else:
        save_name = r"images/naive_kmeans_{0}.png".format(current_iter)
    plt.savefig(save_name)
    plt.clf()


def gen_data(k, n_samples):
    """
    Generates random integer data.
    :param k: number of centers.
    :param n_samples: number of samples.
    :return: random integer data.
    """
    data, y = make_blobs(n_samples=n_samples, centers=k, cluster_std=10, center_box=[0, MAX_DATA], random_state=3)
    data = np.rint(data).astype(int)
    return data
