import secrets


def dot(K, L):
    """
    Performs dot product.
    :param K: First vector.
    :param L:  Second vector.
    :return: Dot product of K and L.
    """
    return sum(i[0] * i[1] for i in zip(K, L))


def ssp(pubkey, prikey, n, X, Y, mult):
    """
    Performs secure scalar product protocol defined by Geothals et al. protocol #3. Can be adapted to N-party using protocol 4.
    https://www.researchgate.net/publication/220833944_On_Private_Scalar_Product_Computation_for_Privacy-Preserving_Data_Mining
    :param pubkey: public key of Alice using Pailliers cryptosystem.
    :param prikey: private key of Alice using Pailliers cryptosystem.
    :param n: modulo.
    :param X: Secret vector of Alice.
    :param Y: Secret vector of Bob.
    :param mult: Integer to multiply vectors with due to expanding of equations.
    :return: Dot product of X and Y split into random shares of Alice (sa) and Bob (sb).
    """
    X = [(int(x) * mult) % n for x in X]
    Y = [(int(y)) % n for y in Y]

    c = []
    N = len(X)
    for i in range(N):
        c.append((pubkey.raw_encrypt(X[i], r_value=25)))  # r value must be gcd(r, n) = 1 for pailliers cryptosystem

    w = (c[0] ** Y[0])
    for i in range(1, N):
        w *= (c[i] ** Y[i])

    sb = secrets.randbelow(n)
    w *= pubkey.raw_encrypt(-sb, r_value=3)
    sa = prikey.raw_decrypt(w)

    return sa, sb
