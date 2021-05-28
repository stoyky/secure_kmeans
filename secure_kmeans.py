import numpy as np
import random
import matplotlib.pyplot as plt
from phe import paillier

from random_share import generate_share, reconstruct_share
from ssp import ssp
from yaos import generate_p_bits_cc, closestcluster
from calculate_terms import calculate_alice_term1, calculate_term2, calculate_bob_term1, calculate_term3

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


def secure_kmeans(data, alice_data, bob_data, k=3, epsilon=0.1, max_iter=1):
    centroids = random_centroids(k)

    for i in range(max_iter):
        # calculating the distance between point and centroid.
        point_center = []

        for idx in range(data.shape[0]):  # iterates 100 times.
            alice_owns_feature1, alice_owns_feature2 = idx_owner(idx, alice_data)
            temp_dist = []

            for centroid in centroids:
                temp_dist.append(dist_euclid(data[idx][0], data[idx][1], centroid[0], centroid[1]))
            point_center.append([np.argmin(temp_dist), [data[idx][0], data[idx][1]]])

            alice_shares = []  # shares after computing the SSP (calculating the 6 terms.)
            bob_shares = []
            for x, y in centroids:
                alice_x, bob_x = generate_share(x, n)
                alice_y, bob_y = generate_share(y, n)

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

                # for checking later. TODO delete
                # g = np.sqrt((alice_complete_term + bob_complete_term) % n)
                # h=  dist_euclid(data[idx][0], data[idx][1], x, y)

                alice_shares.append(alice_complete_term)
                bob_shares.append(bob_complete_term)

            p_bits_cc, input_p_bits_cc = generate_p_bits_cc()  # Get the p_bits for the circuits and inputs for closest cluster
            closest = closestcluster(alice_shares, bob_shares, p_bits_cc,
                                     input_p_bits_cc)  # Returns the index of the smallest sum, i.e. closest cluster
            print("Closest centroid: {0} | Yao's closest centroid: {1}".format(point_center, closest))


if __name__ == '__main__':
    data = np.random.randint(MAX_DATA, size=(100, NUM_FEATURES))
    a, b = create_random_data(
        data)  # receive alice and bob's share for the data for each feature. len(a) = 2 (2d array data)

    secure_kmeans(data, a, b)
