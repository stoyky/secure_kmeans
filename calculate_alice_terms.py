def calculate_alice_term1(data, alice_owns_feature1, alice_owns_feature2):
    if alice_owns_feature1 and alice_owns_feature2:
        return data[0] ** 2 + data[1] ** 2

    elif alice_owns_feature1:
        return data[0] ** 2
    elif alice_owns_feature2:
        return data[1] ** 2
    else:
        return 0


def calculate_term2(centroids):  # list of shares of centroids she owns for all features.
    summation = 0
    for centroid in centroids:
        summation += (centroid **2)

    return summation
