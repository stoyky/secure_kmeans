import numpy as np
import random

### Random shares stuff
from scipy.interpolate import lagrange


def gen_poly(s, n):
    coeff = np.random.randint(0, 10, n - 1)
    poly = [s]
    for a_i in coeff:
        poly.append(a_i * random.randint(0, 100))
    return poly


def eval_poly(poly, x):
    result = 0
    for i in range(0, len(poly)):
        result += poly[i] * x ** i
    return result


def reconstruct(shares):
    shares = np.array(shares)
    lagrange_poly = lagrange(shares[:, 0], shares[:, 1])
    return int(round(lagrange_poly[0]))


def gen_shares(s, n):
    poly = gen_poly(s, n)
    shares = [[i, eval_poly(poly, i)] for i in range(1, n + 1)]
    # shares = [eval_poly(poly, i) for i in range(1, n+1)]
    return shares


def extract_shares(share):  # input is a list [[1 (idx), element_value], [2 (idx), element_value]]. returns element value.
    return share[0][1], share[1][1]
