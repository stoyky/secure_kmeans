import numpy as np
import secrets
import phe.paillier as paillier
import phe.util as pheu


def dot(K, L):
    return sum(i[0] * i[1] for i in zip(K, L))



def ssp(pubkey, prikey, n, X, Y, mult):

    X = [(int(x)*mult) % n for x in X]
    Y = [(int(y)) % n for y in Y]

    # Alice creates C.
    c = []
    N = len(X)
    for i in range(N):
      c.append((pubkey.raw_encrypt(X[i], r_value=25)))  #r value must be gcd(r, n) = 1

    w = (c[0] ** Y[0])
    for i in range(1, N):
        w *= (c[i] ** Y[i])

    sb = secrets.randbelow(n)
    w *= pubkey.raw_encrypt(-sb, r_value=3)
    sa = prikey.raw_decrypt(w)

    return sa, sb
