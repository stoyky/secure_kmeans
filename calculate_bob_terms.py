def calculate_bob_term1(data, alice_owns_feature1, alice_owns_feature2):
    if not alice_owns_feature1 and not alice_owns_feature2:
        return data[0] ** 2 + data[1] ** 2
    elif not alice_owns_feature1:
        return data[0] ** 2
    elif not alice_owns_feature2:
        return data[1] ** 2
    else:
        return 0


def calculate_term3(centroids):  # list of shares of centroids he owns for all features.
    summation = 0
    for centroid in centroids:
        summation += (centroid ** 2)

    return summation
