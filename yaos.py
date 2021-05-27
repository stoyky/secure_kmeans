import numpy as np
import random


# Initializes N, e and d for the oblivious transfer
def initialize_RSA():
    primes = [i for i in range(2, 1000) if checkprime(i)]
    p = random.choice(primes)
    q = random.choice(primes)
    N = p * q
    lN = int(np.lcm(p - 1, q - 1))
    e = random.choice(primes)
    while e >= lN or np.gcd(e, lN) != 1:
        e = random.choice(primes)
    d = pow(e, -1, lN)
    return N, e, d


# Makes it so that Bob can receive the secret value corresponding to his input bit, as only Alice has the p_bits
def obl_transfer(secret_value0, secret_value1, bobs_bit, N, e, d):
    # Bob knows e, Alice's public key. He wants m_c where c is bobs_bit (either 0 or 1)
    # Alice generates a random x_0 and x_1 and sends them to Bob
    x_0 = random.randint(100, 200)
    x_1 = random.randint(100, 200)

    # Bob picks a random k that only he knows
    k = random.randint(100, 1000)
    # He then computes v and sends it to Alice
    if bobs_bit == 0:
        v = x_0 + k**e % N
    else:
        v = x_1 + k**e % N

    # Alice computes k0 and k1 which she uses to mask secret_value0 and secret_value1. She then sends m_0 and m_1 to Bob
    k_0 = (v - x_0)**d % N
    k_1 = (v - x_1)**d % N
    m_0 = secret_value0 + k_0
    m_1 = secret_value1 + k_1

    # Bob computes m_c = m_c' - k and learns nothing about the other m! Should match secret_valuec
    if bobs_bit == 0:
        bobs_new_bit = m_0 - k
    else:
        bobs_new_bit = m_1 - k
    return bobs_new_bit


# Prints the table as seen in the article: each gate has a table showing its output depending on the input
def printtable(gatetype, gatenr, p_bits_gate, p_bits_input_one, p_bits_input_two):
    print("inputs | output | encoding")
    for j in range(2):
        for k in range(2):
            if gatetype == "and":
                print(j, k, " | ", (j and k), " | ", p_bits_gate[(j and k)], "+ hash(", gatenr, ",",
                      p_bits_input_one[j], ",", p_bits_input_two[k], ")")
            if gatetype == "xor":
                print(j, k, " | ", (j ^ k), " | ", p_bits_gate[(j ^ k)], "+ hash(", gatenr, ",", p_bits_input_one[j],
                      ",", p_bits_input_two[k], ")")
            if gatetype == "xnor":
                print(j, k, " | ", (1 - j ^ k), " | ", p_bits_gate[(1 - j ^ k)], "+ hash(", gatenr, ",",
                      p_bits_input_one[j], ",", p_bits_input_two[k], ")")


# Makes sure a and b are of equal length
def padzeros(a, b):
    if len(a) != len(b):
        while len(a) > len(b):
            b.insert(0, 0)
        while len(b) > len(a):
            a.insert(0, 0)
    return a, b


# Create the circuit gates
def add(a, b, p_bits, input_p_bits):
    c = 0
    s = []
    print(a)
    print(b)
    for j in range(len(a) - 1, -1, -1):
        s.append(c ^ a[j] ^ b[j])
        c = c ^ ((a[j] ^ c) and (b[j] ^ c))
    if c == 1:
        s.append(c)
    s.reverse()
    printtable("xor", 1, p_bits[0], input_p_bits[0], [0, 1])  # a ^ c
    printtable("xor", 2, p_bits[1], input_p_bits[1], [0, 1])  # b ^ c
    printtable("and", 3, p_bits[2], p_bits[0], p_bits[1])  # (a ^ c) and (b ^ c)
    printtable("xor", 4, p_bits[3], [0, 1], p_bits[2])  # c ^ ((a ^ c) and (b ^ c))
    printtable("xor", 5, [0, 1], input_p_bits[0], p_bits[1])  # a ^ (b ^ c) is an output gate
    return s


def subtract(a, b, p_bits, input_p_bits):  # Subtract b from a
    c = 1
    s = []
    for j in range(len(a) - 1, -1, -1):
        s.append(c ^ a[j] ^ (1 - b[j]))
        c = c ^ ((a[j] ^ c) and 1 - (b[j] ^ c))
    s.reverse()
    # printtable("xor", 1, p_bits[0], input_p_bits[0], [0, 1])  # a ^ c
    # printtable("xnor", 2, p_bits[1], input_p_bits[1], [0, 1])  # 1 - (b ^ c)
    # printtable("and", 3, p_bits[2], p_bits[0], p_bits[1])  # (a ^ c) and (1 - (b ^ c))
    # printtable("xor", 4, p_bits[3], [0, 1], p_bits[2])  # c ^ ((a ^ c) and (1 - (b ^ c)))
    # printtable("xor", 5, [0, 1], input_p_bits[0], p_bits[1])  # a ^ (1 - (b ^ c)) is an output gate
    return s


def findbiggest(a, b, p_bits, input_p_bits):  # Returns the biggest
    # printtable("xor", 1, p_bits[0], input_p_bits[0], [0, 1])  # a ^ c
    # printtable("xnor", 2, p_bits[1], input_p_bits[1], [0, 1])  # 1 - (b ^ c)
    # printtable("and", 3, p_bits[2], p_bits[0], p_bits[1])  # (a ^ c) and (1 - (b ^ c))
    # printtable("xor", 4, [0, 1], [0, 1], p_bits[2])  # c ^ ((a ^ c) and (1 - (b ^ c))) is an output gate
    c = 0
    for j in range(len(a) - 1, -1, -1):
        c = c ^ ((a[j] ^ c) and 1 - (b[j] ^ c))
    if c == 1:
        return a
    return b


def findsmallest(a, b, p_bits, input_p_bits):  # Returns the smallest
    # printtable("xor", 1, p_bits[0], input_p_bits[0], [0, 1])  # a ^ c
    # printtable("xnor", 2, p_bits[1], input_p_bits[1], [0, 1])  # 1 - (b ^ c)
    # printtable("and", 3, p_bits[2], p_bits[0], p_bits[1])  # (a ^ c) and (1 - (b ^ c))
    # printtable("xor", 4, [0, 1], [0, 1], p_bits[2])  # c ^ ((a ^ c) and (1 - (b ^ c))) is an output gate
    c = 0
    for j in range(len(a) - 1, -1, -1):
        c = c ^ ((a[j] ^ c) and 1 - (b[j] ^ c))
    if c == 1:
        return b
    return a


def division(a, b, p_bits, input_p_bits):  # Returns a / b. Still need to sketch a circuit for this..
    answer = []
    asub = []
    for j in range(len(a)):
        asub.append(a[j])
        asub, b = padzeros(asub, b)
        if findbiggest(asub, b, p_bits[:4], input_p_bits[:2]) == asub:  # Need to write a gate that checks for equality?
            answer.append(1)
            asub = subtract(asub, b, p_bits[4:], input_p_bits[2:])
        elif findbiggest(asub, b, p_bits[:4], input_p_bits[:2]) == b:
            answer.append(0)
    return answer


# Create the circuits
def closestcluster(x, y, p_bits, input_p_bits):  # Returns the index of the smallest sum
    summation = []
    smallest = 0
    for k in range(len(x)):  # First k circuits
        a = [int(j) for j in bin(x[k])[2:]]  # Transform each element of x and y to a binary array
        b = [int(j) for j in bin(y[k])[2:]]
        a, b = padzeros(a, b)
        summation.append(add(a, b, p_bits[:4], input_p_bits[:2]))
    for k in range(len(x) - 1):  # Second k circuits
        summation[smallest], summation[k + 1] = padzeros(summation[smallest], summation[k + 1])
        small = findsmallest(summation[smallest], summation[k + 1], p_bits[4:], input_p_bits[2:])
        if small == summation[k + 1]:
            smallest = k + 1
    return smallest


def recomputemean(a, b, m, n, p_bits, input_p_bits):  # Returns (a+b)/(m+n)
    a = [int(x) for x in bin(a)[2:]]
    b = [int(x) for x in bin(b)[2:]]
    m = [int(x) for x in bin(m)[2:]]
    n = [int(x) for x in bin(n)[2:]]
    a, b = padzeros(a, b)
    m, n = padzeros(m, n)
    ab = add(a, b, p_bits[:4], input_p_bits[:2])  # First circuit
    mn = add(m, n, p_bits[:4], input_p_bits[2:4])  # Second circuit
    ab, mn = padzeros(ab, mn)
    answer = division(ab, mn, p_bits[4:], input_p_bits[4:])  # Remaining circuits
    return answer


def checkprime(n):
    for i in range(2, n):
        if (n % i) == 0:
            return False
    return True


def generate_p_bits_cc():
    # Now we need p_bits (randomly generated labels) to represent the input values (0 or 1).
    # Each gate that is not an output gate also needs these labels as they are used as input for another gate.
    p_bits_cc = np.random.randint(0, 100, (7, 2))  # 7 values; addition(4) and findsmallest(3)
    # Structure: [firstgate0add, firstgate1add],[secondgate0add, secondgate1add], etc.
    input_p_bits_cc = np.random.randint(0, 100, (4, 2))  # 2 sets for addition, 2 sets for findsmallest
    # Structure: [a0add, a1add],[b0add, b1add], [a0fs, a1fs, b0fs, b1fs]
    return p_bits_cc, input_p_bits_cc


def generate_p_bits_rm():
    p_bits_rm = np.random.randint(0, 100, (11, 2))  # 11 values; addition(4) and division(findbiggest(3) and subtract(4))
    input_p_bits_rm = np.random.randint(0, 100, (8, 2))  # 4 sets for addition, 4 sets for findbiggest
    return p_bits_rm, input_p_bits_rm


if __name__ == '__main__':
    alice = [3, 1, 2, 2, 4]
    bob = [3, 4, 5, 2, 1]
    N, e, d = initialize_RSA()  # Initiate RSA for oblivious transfer
    p_bits_cc, input_p_bits_cc = generate_p_bits_cc()  # Get the p_bits for the circuits and inputs for closest cluster
    p_bits_rm, input_p_bits_rm = generate_p_bits_rm()  # Get the p_bits for the circuits and inputs for recompute mean
    closest = closestcluster(alice, bob, p_bits_cc, input_p_bits_cc)  # Returns the index of the smallest sum, i.e. closest cluster
    print(closest)
