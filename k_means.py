import numpy as np
from matplotlib import pyplot as plt

def dist_euclid(x1, y1, x2, y2):
  return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def kmeans(D, k, max_iter, epsilon):
  centroids = []

  # randomly initialize centroids
  for i in range(k):
    centroids.append([np.random.randint(0, 10), np.random.randint(0, 10)])

  for i in range(max_iter):
  # calculate distance from each point to each centroid and choose minimum
    point_center = []
    for point in D:
      temp_dist = []
      for centroid in centroids:
          temp_dist.append(dist_euclid(point[0], point[1], centroid[0], centroid[1]))
      point_center.append([ np.argmin(temp_dist),[point[0], point[1]]])

    result = {}

    for i in range(0,k):
      temp = []
      for point in point_center:
        if point[0] == i:
          temp.append(point[1])
      result[i] = temp

    plt.figure(figsize=(4,4))
    plt.title("")
    colors = ["red", "blue", "green"]
    for i in range(0, k):
      plt.scatter(centroids[i][0], centroids[i][1], color="black",sizes=[100.0], marker='X', zorder=1)
      for point in result[i]:
        plt.scatter(point[0], point[1], sizes=[10.0], color=colors[i], zorder=0)

    # need to check if centroids differ by no more than epsilon and then just terminate
    centroids_avg = []
    for i in range(0, k):
      centroids_avg.append(np.average(result[i], axis=0))

    centroids = centroids_avg

kmeans(D, k, 10, 0.01)


