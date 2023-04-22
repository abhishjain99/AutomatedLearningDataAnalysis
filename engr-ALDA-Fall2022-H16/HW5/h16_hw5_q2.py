import pandas as pd
import numpy as np
import math
from tabulate import tabulate
from sklearn.cluster import KMeans

arr_c = {'A':[5,-3], 'B':[-1,5], 'C':[2,1], 'D':[-4,-4], 'E':[4,5], 'F':[1,-2], 'G':[-3,6], 'H':[4,4], 'I':[3,-1], 'J':[3,-2], 'K':[3,-3]}
arr = [[5,-3],[-1,5],[2,1],[-4,-4],[4,5],[1,-2],[-3,6],[4,4],[3,-1],[3,-2],[3,-3]]
arr_lab = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

def find_centroids(clus0, clus1, clus2):
    centroids = []
    centroids.append([np.mean([arr_c[k][0] for k in clus0]), np.mean([arr_c[k][1] for k in clus0])])
    centroids.append([np.mean([arr_c[k][0] for k in clus1]), np.mean([arr_c[k][1] for k in clus1])])
    centroids.append([np.mean([arr_c[k][0] for k in clus2]), np.mean([arr_c[k][1] for k in clus2])])
    return centroids

def find_answer(centroids, typ):
    arr0 = []
    arr1 = []
    arr2 = []
    for p in clus0:
        arr0.append(math.dist(arr_c[p],centroids[0])**2)
    for p in clus1:
        arr1.append(math.dist(arr_c[p],centroids[1])**2)
    for p in clus2:
        arr2.append(math.dist(arr_c[p],centroids[2])**2)

    df = pd.DataFrame([arr0,clus0])
    print('Cluster 0')
    print(tabulate(df, headers = 'keys', tablefmt = "fancy_grid"))

    df = pd.DataFrame([arr1,clus1])
    print('Cluster 1')
    print(tabulate(df, headers = 'keys', tablefmt = "fancy_grid"))

    df = pd.DataFrame([arr2,clus2])
    print('Cluster 2')
    print(tabulate(df, headers = 'keys', tablefmt = "fancy_grid"))
    sum_arr0 = np.sum(arr0)
    sum_arr1 = np.sum(arr1)
    sum_arr2 = np.sum(arr2)
    print(f'--- SSE {typ} ---')
    print('Cluster 0:', sum_arr0)
    print('Cluster 1:', sum_arr1)
    print('Cluster 2:', sum_arr2)
    print('Total:', sum_arr0+sum_arr1+sum_arr2)

# ###### S L H C ###### #
clus0 = ['E', 'H', 'I', 'J', 'K', 'A', 'F', 'C']
clus1 = ['B', 'G']
clus2 = ['D']

centroids = find_centroids(clus0, clus1, clus2)
print('single centroids', centroids)
find_answer(centroids, 'single link')

# ###### C L H C ###### #
clus0 = ['B', 'E', 'G', 'H']
clus1 = ['A', 'C', 'F', 'I', 'J', 'K']
clus2 = ['D']

centroids = find_centroids(clus0, clus1, clus2)
print('compete centroids', centroids)
find_answer(centroids, 'complete link')

# ###### K M C ###### #
clus0 = ['B', 'E', 'G', 'H']
clus1 = ['C', 'F', 'D']
clus2 = ['A', 'I', 'J', 'K']

centroids = find_centroids(clus0, clus1, clus2)
print('compete centroids', centroids)
find_answer(centroids, 'complete link')