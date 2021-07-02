def calculate_p0_term1(data, p0_owns_feature1, p0_owns_feature2, n):
    if p0_owns_feature1 and p0_owns_feature2:
        return (data[0] ** 2 + data[1] ** 2) % n

    elif p0_owns_feature1:
        return (data[0] ** 2) % n
    elif p0_owns_feature2:
        return (data[1] ** 2) % n
    else:
        return 0


def calculate_p1_term1(data, p0_owns_feature1, p0_owns_feature2, n):
    if not p0_owns_feature1 and not p0_owns_feature2:
        return (data[0] ** 2 + data[1] ** 2) % n
    elif not p0_owns_feature1:
        return (data[0] ** 2) % n
    elif not p0_owns_feature2:
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
