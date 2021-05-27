import numpy as np
import secrets
import phe.paillier as paillier
import phe.util as pheu

def dot(K, L):
    return sum(i[0] * i[1] for i in zip(K, L))

def ssp(pubkey, prikey, n, X, Y):

    X = [int(x) for x in X]
    Y = [int(y) for y in Y]

    xy = np.dot(X, Y)

    # Alice creates C.
    c = []
    N = len(X)
    for i in range(N):
      c.append((pubkey.raw_encrypt(X[i], r_value=25)))  #r value must be gcd(r, n) = 1

    w = (c[0] ** Y[0])

    for i in range(1, N):
        w *= (c[i] ** Y[i])

    sb = secrets.randbelow(n)
    t = pubkey.raw_encrypt(-sb, r_value=3)
    w = (t * w)
    sa = prikey.raw_decrypt(w)

    # truth = (dot(X, Y) - sb) % n
    # print("n: {4} | sa: {0} | sb: {1} | dot product: {2} | true sa: {3}".format(sa, sb, dot(X, Y), truth, n))

    return sa, sb
