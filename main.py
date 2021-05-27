import numpy as np
import random

from scipy.interpolate import lagrange

def gen_point_shares(point, n):
  x_shares = gen_shares(point[0], n)
  y_shares = gen_shares(point[1], n)
  return x_shares, y_shares


def distribute_centroids(centroids, n):
    result = {}
    for i in range(0, num_parties-1):
      x_ = []
      y_ = []

      for centroid in centroids:
        x_shares, y_shares = gen_point_shares(centroid, n)
        for x,y in x_shares, y_shares:
          x_.append(x)
          y_.append(y)

      result[i] = x_
      result[i+1] = y_

    return result

def reconstruct_centroids(shares):
  elem = []
  for i in range(0, num_parties * num_clusters):
    share_elem = []
    for j in range(0, num_parties):
      share_elem.append(shares[j][i])
    elem.append(reconstruct(share_elem))

  result = []
  for i in range(1, num_parties* num_clusters):
    if i % 2 != 0:
      result.append([elem[i-1], elem[i]])

  return result



def share_to_vector(share):
    result = []
    for i in range(0, len(share)):
        result.append(share[i][1])
    return result


def secure_closest_cluster(d_i, centroids):
    print(d_i)


if __name__ == '__main__':


    k = 3
    num_parties = 2
    shares = {}

    D = [[2, 3], [4, 5], [6, 7]]

    # "Randomly" select k objects from D as initial cluster centers
    centroids = [[2, 3], [4, 5], [6, 7]]

    # "Randomly" share cluster centers between Alice and Bob
    a_shares = []
    b_shares = []
    for coord in centroids:
        for feat in coord:
            shares = gen_shares(feat, 2)
            a_shares.append(shares[0])
            b_shares.append(shares[1])

    # Repeat
    # while epsilon too large
    mu_a = a_shares
    mu_b = b_shares

    sum = 0
    for j in range(k):
        for d_i in D:
            for m in range(len(d_i)):
                x_i = d_i[m]
                sum += (x_i - (mu_a[m] + mu_b[m])) ** 2
                print(str(x_i) + " - (" + str(mu_a[m]) + " + " + str(mu_b[m]) + ") ----- " + str(
                    reconstruct([[1, mu_a[m]], [2, mu_b[m]]])))

    print(sum)

# print(shares)
# print(parties)
# combined_x = [a_shares[0], b_shares[0]]
# combined_y = [a_shares[1], b_shares[1]]

# print("combined x: " + str(reconstruct(combined_x)))
# print("combined y: " + str(reconstruct(combined_y)))











