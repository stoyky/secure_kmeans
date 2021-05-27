import numpy as np
import random
import matplotlib.pyplot as plt
from phe import paillier

from random_share import gen_shares, reconstruct, extract_shares
from ssp import ssp
from yaos import generate_p_bits_cc, closestcluster
from calculate_alice_terms import calculate_alice_term1, calculate_term2
from calculate_bob_terms import calculate_bob_term1, calculate_term3

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


def secure_kmeans(data, alice_data, bob_data, k=3, epsilon=0.1, max_iter=10):
    centroids = random_centroids(k)


    for i in range(max_iter):
        # calculating the distance between point and centroid.
        for idx in range(data.shape[0]):  # iterates 100 times.
            alice_owns_feature1, alice_owns_feature2 = idx_owner(idx, alice_data)

            alice_shares = []  # shares after computing the SSP (calculating the 6 terms.)
            bob_shares = []
            for x, y in centroids:
                alice_x, bob_x = extract_shares(gen_shares(x, NUM_FEATURES))
                alice_y, bob_y = extract_shares(gen_shares(y, NUM_FEATURES))

                alice_term1 = calculate_alice_term1(data[idx], alice_owns_feature1, alice_owns_feature2)
                bob_term1 = calculate_bob_term1(data[idx], alice_owns_feature1, alice_owns_feature2)

                alice_term2 = calculate_term2([alice_x, alice_y])  # calculate the summation of her centroid shares.
                bob_term3 = calculate_term3([bob_x, bob_y])  # calculate the summation

                sa, sb = ssp(pubkey, prikey, n, [alice_term1, alice_term2], [bob_term1, bob_term3])
                alice_shares.append(sa)
                bob_shares.append(sb)


if __name__ == '__main__':
    data = np.random.randint(MAX_DATA, size=(100, NUM_FEATURES))
    a, b = create_random_data(
        data)  # receive alice and bob's share for the data for each feature. len(a) = 2 (2d array data)

    secure_kmeans(data, a, b)
