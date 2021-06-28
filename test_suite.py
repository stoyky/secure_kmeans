import unittest
from random_share import *
from ssp import *
from yaos import *


class TestRandomShare(unittest.TestCase):
    def test_reconstruction(self):
        shares = gen_shares(1234, 2)
        self.assertEqual(1234, reconstruct(shares))


class TestSSP(unittest.TestCase):
    def test_ssp(self):
        X = [1, 2, 3]
        Y = [3, 4, 5]

        Z = [(-2) * x for x in X]

        self.assertEqual((-2) * dot(X, Y), dot(Z, Y))


class TestYaos(unittest.TestCase):
    def test_yaos(self):
        alice = [3, 1, 2, 2, 4]
        bob = [3, 4, 5, 2, 1]
        N, e, d = initialize_RSA()
        p_bits_cc, input_p_bits_cc = generate_p_bits_cc()
        p_bits_rm, input_p_bits_rm = generate_p_bits_rm()
        p = 347
        q = 349
        n = p * q
        closest = closestcluster(alice, bob, p_bits_cc, input_p_bits_cc,n)
        self.assertEqual(closest, 3)


if __name__ == '__main__':
    unittest.main()