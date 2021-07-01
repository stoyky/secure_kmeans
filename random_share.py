import numpy as np
import random
from scipy.interpolate import lagrange


def gen_poly(s, n):
    """
    Generate polynomial from s + n_0 + n_1 + ... + n-1
    where s is the secret and
    n is the number of parties over which we wish to distribute
    :param s: secret (int)
    :param n: number of parties (int)
    :return: polynomial as list of integers
    """
    coeff = np.random.randint(0, 10, n - 1)
    poly = [s]
    for a_i in coeff:
        poly.append(a_i * random.randint(0, 100))
    return poly


def eval_poly(poly, x):
    """
    Evaluates polynomial at point x
    :param poly: polynomial as list of integers
    :param x: point to evaluate on poly
    :return: result of evaluation (int)
    """
    result = 0
    for i in range(0, len(poly)):
        result += poly[i] * x ** i
    return result


def reconstruct(shares):
    """
    Reconstructs the secret from the shares
    :param shares: list of shares
    :return: (int) the secret that was distributed
    """
    shares = np.array(shares)
    lagrange_poly = lagrange(shares[:, 0], shares[:, 1])
    return int(round(lagrange_poly[0]))


def gen_shares(s, n):
    """
    Generates n shares to share the secret integer s
    :param s: secret integer to share
    :param n: number of parties to create shares for
    :return: list of random shares
    """
    poly = gen_poly(s, n)
    shares = [[i, eval_poly(poly, i)] for i in range(1, n + 1)]
    return shares


def naive_extract_shares(share):
    """
    Utility function to extract element values from the shares
    :param share: shares from which to extract element
    :return: tuple of extracted elements
    """
    return share[0][1], share[1][1]


def naive_generate_share(x, n):
    """
    Naive sharing of a secret, inspired by Jagannathan and Wright's papers
    :param x: secret to share
    :param n: modulus
    :return: tuple of share and random value that was used to share
    """
    random_value = (np.random.randint(1000) % n)
    share1 = (x - random_value) % n

    return share1, random_value


def naive_reconstruct_share(x, y, n):
    """
    Naive share reconstruction
    :param x: share x (int)
    :param y: share y (int)
    :param n: modulus
    :return: reconstructed secret (int)
    """
    return (x + y) % n
