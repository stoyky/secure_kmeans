from copy import deepcopy
import matplotlib.pyplot as plt
from phe import paillier
from random_share import generate_share
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
    alice_data = []
    bob_data = []

    for column_idx in range(data.shape[1]):
        random_value = random.randint(0, data.shape[0])  # choose random value between 0 - 100.

        # randomly select indices for A of random value amount.
        possible_index = list(range(data.shape[0]))
        alice_idx = []  # index of alice owned elements of column X.

        for i in range(random_value):
            random_index = random.choice(possible_index)
            possible_index.remove(random_index)
            alice_idx.append(random_index)

        alice_data.append(sorted(alice_idx))
        bob_idx = list(set(range(data.shape[0])) - set(alice_idx))
        bob_data.append(sorted(bob_idx))

    return alice_data, bob_data


def idx_owner(idx, alice_data):
    feature1_owner_alice = False
    feature2_owner_alice = False

    if idx in alice_data[0]:
        feature1_owner_alice = True

    if idx in alice_data[1]:
        feature2_owner_alice = True

    return feature1_owner_alice, feature2_owner_alice


def random_centroids(k):
    centroids = []
    for i in range(k):
        centroids.append([np.random.randint(0, MAX_DATA), np.random.randint(0, MAX_DATA)])

    return centroids


def dist_euclid(x1, y1, x2, y2):
    return np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2) % n)


def secure_kmeans(data, centroids, k, epsilon, max_iter, plot=False):
    alice_data, bob_data = create_random_data(data)
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
                alice_owns_feature1, alice_owns_feature2 = idx_owner(idx, alice_data)
                alice_shares = []  # shares after computing the SSP (calculating the 6 terms.)
                bob_shares = []

                alice_old_centroids = []
                bob_old_centroids = []
                old_centroids = []
                for x, y in centroids:
                    alice_x, bob_x = generate_share(x, n)
                    alice_y, bob_y = generate_share(y, n)

                    old_centroids = deepcopy(centroids)  # TODO DELETE.
                    alice_old_centroids.append((alice_x, alice_y))
                    bob_old_centroids.append((bob_x, bob_y))
                    # reconstruct_share(alice_x, bob_x, n)
                    # reconstruct_share(alice_y, bob_y, n)

                    alice_term1 = calculate_alice_term1(data[idx], alice_owns_feature1, alice_owns_feature2, n)
                    bob_term1 = calculate_bob_term1(data[idx], alice_owns_feature1, alice_owns_feature2, n)

                    alice_term2 = calculate_term2([alice_x, alice_y], n)  # calculate the summation of her centroid shares.
                    bob_term3 = calculate_term3([bob_x, bob_y], n)  # calculate the summation

                    alice_term4, bob_term4 = ssp(pubkey, prikey, n, [alice_x, alice_y], [bob_x, bob_y], mult=2)
                    alice_term5, bob_term5 = ssp(pubkey, prikey, n, [alice_x, alice_y], data[idx].tolist(), mult=2)
                    alice_term6, bob_term6 = ssp(pubkey, prikey, n, data[idx].tolist(), [bob_x, bob_y], mult=2)
                    # print((alice_term5 + bob_term5) % n)
                    # print((alice_term4 + bob_term4) % n)
                    # print((alice_term6 + bob_term6) % n)

                    alice_complete_term = (alice_term1 + alice_term2 + alice_term4 - alice_term5 - alice_term6) % n
                    bob_complete_term = (bob_term1 + bob_term3 + bob_term4 - bob_term5 - bob_term6) % n

                    # for checking later. TODO delete debug statement
                    g = np.sqrt((alice_complete_term + bob_complete_term) % n)
                    h = dist_euclid(data[idx][0], data[idx][1], x, y)
                    #
                    if g != h:
                        print("error in calculating distance")

                    alice_shares.append(alice_complete_term)
                    bob_shares.append(bob_complete_term)

                p_bits_cc, input_p_bits_cc = generate_p_bits_cc()  # Get the p_bits for the circuits and inputs for closest cluster
                closest = closestcluster(alice_shares, bob_shares, p_bits_cc,
                                         input_p_bits_cc,
                                         n)  # Returns the index of the smallest sum, i.e. closest cluster
                closest_cluster.append(closest)  # add the closest cluster for the data point.

            centroids_avg = []
            # 2. recompute the mean
            for _k in range(k):
                data_point_indicies = np.where(np.asarray(closest_cluster) == _k)[0]

                alice_sum = [0, 0]  # track of alice's summation for current cluster centroids for data point she owns. (
                # (feature1, feature2)
                alice_owns_count = [0, 0]  # track of number of data points she owns in current cluster centroid.
                bob_sum = [0, 0]
                bob_owns_count = [0, 0]

                for data_point_index in data_point_indicies:
                    alice_owns_feature1, alice_owns_feature2 = idx_owner(data_point_index, alice_data)

                    if alice_owns_feature1:
                        alice_sum[0] += data[data_point_index][0]
                    else:
                        bob_sum[0] += data[data_point_index][0]

                    if alice_owns_feature2:
                        alice_sum[1] += data[data_point_index][1]
                    else:
                        bob_sum[1] += data[data_point_index][1]

                    if alice_owns_feature1 and alice_owns_feature2:  # alice owns everything. # 1 1
                        alice_owns_count = [count + 1 for count in alice_owns_count]
                    elif (not alice_owns_feature1) and (not alice_owns_feature2):  # bob owns everything # 0 0
                        bob_owns_count = [count + 1 for count in bob_owns_count]
                    elif alice_owns_feature1:  # 1 0
                        alice_owns_count[0] += 1
                        bob_owns_count[1] += 1
                    else:  # 0 1
                        alice_owns_count[1] += 1
                        bob_owns_count[0] += 1

                p_bits_rm, input_p_bits_rm = generate_p_bits_rm()

                for j in range(NUM_FEATURES):
                    centroids[_k][j] = recomputemean(alice_sum[j], bob_sum[j], alice_owns_count[j], bob_owns_count[j],
                                              p_bits_rm, input_p_bits_rm)

            # 3. check for termination.
            alices_centroid_shares = []
            bobs_centroid_shares = []
            for centroid, alice_old_centroid, bob_old_centroid in zip(centroids, alice_old_centroids, bob_old_centroids):
                alice_x, bob_x = generate_share(centroid[0], n)
                alice_y, bob_y = generate_share(centroid[1], n)

                # calculate term 1 = old centroids squared.
                # expand brackets = (ua1 ^ 2 + ub1 ^ 2 + 2ua1b1) + (ua2 ^ 2 + ub2 ^ 2 + 2ua2b2)
                a_old_squared = 0
                b_old_squared = 0

                for i in range(NUM_FEATURES):  # iterate over all features.
                    a_old_squared += (alice_old_centroid[i] ** 2) % n
                    b_old_squared += (bob_old_centroid[i] ** 2) % n
                a_share_term1, b_share_term1 = ssp(pubkey, prikey, n, list(alice_old_centroid), list(bob_old_centroid), mult=2)  # calculate 2*uab

                # calculate term 2 = new centroids squared.
                a_new_squared = 0
                b_new_squared = 0

                alice_new_centroid = (alice_x, alice_y)
                bob_new_centroid = (bob_x, bob_y)

                for i in range(NUM_FEATURES):  # iterate over all features.
                    a_new_squared += (alice_new_centroid[i] ** 2) % n
                    b_new_squared += (bob_new_centroid[i] ** 2) % n
                a_share_term2, b_share_term2 = ssp(pubkey, prikey, n, list(alice_new_centroid), list(bob_new_centroid),
                                                   mult=2)  # calculate 2*uab

                # calculate term 3 = (new_a + new_b) (old_a + old_b) +  (new_a + new_b) (old_a + old_b)
                a_old_new = 0  # new_a * old_a
                b_old_new = 0  # new_b * old_b

                for i in range(NUM_FEATURES):
                    a_old_new += (2 * (alice_new_centroid[i] * alice_old_centroid[i])) % n
                    b_old_new += (2 * (bob_new_centroid[i] * bob_old_centroid[i])) % n

                # new_b * old_a
                a_share_term3, b_share_term3 = ssp(pubkey, prikey, n,  list(bob_new_centroid), list(alice_old_centroid),
                                                   mult=2)  # calculate 2*uab

                # new_a * old_b
                a_share_term4, b_share_term4 = ssp(pubkey, prikey, n, list(alice_new_centroid), list(bob_old_centroid),
                                                   mult=2)  # calculate 2*uab
                alice_centroid_term = (a_old_squared + a_share_term1 + a_new_squared +\
                                      a_share_term2 - a_old_new - a_share_term3 - a_share_term4) % n
                bob_centroid_term = (b_old_squared + b_share_term1 + b_new_squared +\
                                      b_share_term2 - b_old_new - b_share_term3 - b_share_term4) % n

                alices_centroid_shares.append(alice_centroid_term)
                bobs_centroid_shares.append(bob_centroid_term)

                # # TODO DELETE.
                # g  = np.sqrt((alice_centroid_term + bob_centroid_term) % n)
                # h = dist_euclid(centroid[0], centroid[1], old_centroids[0][0], old_centroids[0][1])

            p_bits_tm = np.random.randint(0, MAX_DATA, (3, 2))  # 3 values; findbiggest(3)
            input_p_bits_tm = np.random.randint(0, MAX_DATA, (4, 2))  # 4 for findbiggest
            below_epsilon = 1

            for a_share, b_share in zip(alices_centroid_shares, bobs_centroid_shares):
                ab = (a_share + b_share) % n  # epsilon has to be a integer. 0.2 * 10 = 2.
                # returns 0 if any is above epsilon, thus 1*0 = 0. if all is below epsilon, remains 1.
                below_epsilon *= terminate(ab, epsilon, p_bits_tm, input_p_bits_tm)

            converged = bool(below_epsilon)

            # # 4. plot some nice graphs.
            if plot:
                plot_and_save(data, k, centroids, closest_cluster, current_iter, secure=True)

        else:
            print("MAX ITERATIONS REACHED.")
            converged = True  # too many iterations.

    # 5. Allocate cluster centers to Alice and Bob
    alice_cluster_centers = []
    bob_cluster_centers = []
    for _k in range(k):
        data_point_indices = np.where(np.asarray(closest_cluster) == _k)[0]

        for idx in data_point_indices:
            alice_owns_feature1, alice_owns_feature2 = idx_owner(idx, alice_data)
            if alice_owns_feature1 and alice_owns_feature2:  # alice owns all features
                alice_cluster_centers.append(tuple(centroids[_k]))
                # bob owns all features.
            elif not(alice_owns_feature1) and not(alice_owns_feature2):
                bob_cluster_centers.append(tuple(centroids[_k]))
                # they share features
            else:
                alice_cluster_centers.append(tuple(centroids[_k]))
                bob_cluster_centers.append(tuple(centroids[_k]))

    alice_cluster_centers = set(alice_cluster_centers)
    bob_cluster_centers = set(bob_cluster_centers)

    # print("Alice's cluster centers are: {0}".format(alice_cluster_centers))
    # print("Bob's cluster centers are: {0}".format(bob_cluster_centers))
    print("done")


def naive_kmeans(data, centroids, k, epsilon, max_iter, plot=False):
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
    data, y = make_blobs(n_samples=n_samples, centers=k, cluster_std=10, center_box=[0, MAX_DATA], random_state=3)
    data = np.rint(data).astype(int)
    return data
