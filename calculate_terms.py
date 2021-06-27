def calculate_alice_term1(data, alice_owns_feature1, alice_owns_feature2, n):
    if alice_owns_feature1 and alice_owns_feature2:
        return (data[0] ** 2 + data[1] ** 2) % n

    elif alice_owns_feature1:
        return (data[0] ** 2) % n
    elif alice_owns_feature2:
        return (data[1] ** 2) % n
    else:
        return 0


def calculate_bob_term1(data, alice_owns_feature1, alice_owns_feature2, n):
    if not alice_owns_feature1 and not alice_owns_feature2:
        return (data[0] ** 2 + data[1] ** 2) % n
    elif not alice_owns_feature1:
        return (data[0] ** 2) % n
    elif not alice_owns_feature2:
        return (data[1] ** 2) % n
    else:
        return 0


def calculate_term2(centroids, n):  # list of shares of centroids she owns for all features.
    summation = 0
    for centroid in centroids:
        summation += ((centroid ** 2) % n)

    return summation


def calculate_term3(centroids, n):  # list of shares of centroids he owns for all features.
    summation = 0
    for centroid in centroids:
        summation += ((centroid ** 2) % n)

    return summation


