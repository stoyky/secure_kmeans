import numpy as np
import secrets
import phe.paillier as paillier
import phe.util as pheu

def dot(K, L):
    return sum(i[0] * i[1] for i in zip(K, L))

def ssp(pubkey, prikey, n, X, Y, mult=1):
    dot_product = dot(X,Y)

    X = [int(x)*mult for x in X]
    Y = [int(y) for y in Y]

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
    sa = prikey.raw_decrypt(int(w))

    #truth = (mult*dot_product - sb) % n
    #print(" sa: {0} | sb: {1} | 2x dot product: {2} | true sa: {3} | n: {4} ".format(sa, sb, mult*dot_product, truth, n))

    return sa, sb

if __name__ == '__main__':
    X = [1,2,3]
    Y = [3,4,5]

    Z = [(-2)*x for x in X]

    print((-2)*dot(X,Y))
    print(dot(Z, Y))