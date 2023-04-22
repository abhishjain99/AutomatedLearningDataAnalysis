import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

filename = "Algerian_forest_fires_dataset_UPDATE.csv"
# a - data cleaning and scatter plot
clean_df1 = pd.read_csv(filename, skiprows=1, nrows = 122)
clean_df2 = pd.read_csv(filename, skiprows=126, nrows=122)
combined_clean_df = [clean_df1, clean_df2]
concatenated_cleaned_df = pd.concat(combined_clean_df)
final_cleaned_df = concatenated_cleaned_df.filter([" RH", " Ws"])
final_cleaned_df.plot(x =' RH', y=' Ws', title='relative humidity and wind speed', kind='scatter', xlabel='relative humidity', ylabel='wind speed')


# b - defining a data point
RH_mean = final_cleaned_df[" RH"].mean() # 61.93852459016394
Ws_mean = final_cleaned_df[" Ws"].mean() # 15.504098360655737
P = (RH_mean, Ws_mean)
print('Mean Point:', P)
tuple_data = [(a, b) for a, b in final_cleaned_df.values]


# c 1 - Euclidean distance
close_eucledian_points = []
euc_dist = [math.sqrt((P[0] - a[0]) ** 2 + (P[1] - a[1]) ** 2) for a in tuple_data]
close_eucledian_points.append([tuple_data[i] for i in np.argsort(euc_dist)[1:7]])
print('close Eucledian points--', close_eucledian_points)

# c 2 - Manhattan block metric
close_manhattan_pts = []
man_dist = [abs(P[0] - a[0]) + abs(P[1] - a[1]) for a in tuple_data]
close_manhattan_pts.append([tuple_data[i] for i in np.argsort(man_dist)[1:7]])
print('close Manhattan points--', close_manhattan_pts)

# c 3 - Minkowski metric for power = 7
close_minkowski_pts = []
pow_val = 7
minkow_dist = [math.pow((math.pow(abs(P[0] - a[0]), 7) + math.pow(abs(P[1] - a[1]), 7)), 1/7) for a in tuple_data]
close_minkowski_pts.append([tuple_data[i] for i in np.argsort(minkow_dist)[1:7]])
print('close Minkowski points--', close_minkowski_pts)

# c 4 - Chebyshev distance
close_Chebyshev_pts = []
cheby_dist = [max(abs(P[0] - a[0]), abs(P[1] - a[1])) for a in tuple_data]
close_Chebyshev_pts.append([tuple_data[i] for i in np.argsort(cheby_dist)[1:7]])
print('close Chebyshev points-', close_Chebyshev_pts)

# c 5 - Cosine distance
close_Cosine_pts = []
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
cosine_dist = [1 - (P[0] * a[0] + P[1] * a[1]) / (math.sqrt(P[0] ** 2 + P[1] ** 2) * math.sqrt(a[0] ** 2 + a[1] ** 2)) for a in tuple_data]
close_Cosine_pts.append([tuple_data[i] for i in np.argsort(cosine_dist)[1:7]])
print('close Cosine points-', close_Cosine_pts)


# d i - Plots for distance measures
f, ax1 = plt.subplots(2,2)
f.subplots_adjust(hspace=1)
# # d i 1 - Euclidean distance
close_eucledian_points_20 = [tuple_data[i] for i in np.argsort(euc_dist)[1:21]]
for x, y in close_eucledian_points_20:
    ax1[0,0].scatter(x, y, color='green', marker='*', s=20, alpha=0.6)
ax1[0,0].scatter(P[0], P[1], color='magenta', marker='D', s=25, alpha = 1)
ax1[0,0].set_xlabel('Relative Humidity')
ax1[0,0].set_ylabel('Wind Speed')
ax1[0,0].set_title('Eucledian Distance')

# # d i 2 - Manhattan block metric
close_manhattan_pts_20 = [tuple_data[i] for i in np.argsort(man_dist)[1:21]]
for x, y in close_manhattan_pts_20:
    ax1[0,1].scatter(x, y, color='blue', marker='x', s=20, alpha=0.6)
ax1[0,1].scatter(P[0], P[1], color='magenta', marker='D', s=25, alpha = 1)
ax1[0,1].set_xlabel('Relative Humidity')
ax1[0,1].set_ylabel('Wind Speed')
ax1[0,1].set_title('Manhattan Block Metric')

# # d i 3 - Minkowski metric
close_minkowski_pts_20 = [tuple_data[i] for i in np.argsort(minkow_dist)[1:21]]
for x, y in close_minkowski_pts_20:
    ax1[1,0].scatter(x, y, color='red', marker='+', s=20, alpha=0.6)
ax1[1,0].scatter(P[0], P[1], color='magenta', marker='D', s=25, alpha = 1)
ax1[1,0].set_xlabel('Relative Humidity')
ax1[1,0].set_ylabel('Wind Speed')
ax1[1,0].set_title('Minkowski Metric')

# # d i 4 - Chebyshev distance
close_Chebyshev_pts_20 = [tuple_data[i] for i in np.argsort(cheby_dist)[1:21]]
for x, y in close_Chebyshev_pts_20:
    ax1[1,1].scatter(x, y, color='black', marker='o', s=20, alpha=0.6)
ax1[1,1].scatter(P[0], P[1], color='magenta', marker='D', s=25, alpha = 1)
ax1[1,1].set_xlabel('Relative Humidity')
ax1[1,1].set_ylabel('Wind Speed')
ax1[1,1].set_title('Chebyshev Distance')

f, ax2 = plt.subplots(1,1)
# d i 5 - Cosine distance
close_Cosine_pts_20 = [tuple_data[i] for i in np.argsort(cosine_dist)[1:21]]
for x, y in close_Cosine_pts_20:
    ax2.scatter(x, y, color='orange', marker='v', s=20, alpha=0.8)
ax2.scatter(P[0], P[1], color='magenta', marker='D', s=25, alpha = 1)
ax2.set_xlabel('Relative Humidity')
ax2.set_ylabel('Wind Speed')
ax2.set_title('Cosine Distance')

# f, ax3 = plt.subplots(1,1)
# for x, y in close_eucledian_points_20:
#     ax3.scatter(x, y, color='green', marker='*', s=100, alpha=0.3)
# for x, y in close_manhattan_pts_20:
#     ax3.scatter(x, y, color='blue', marker='x', s=80, alpha=0.3)
# for x, y in close_minkowski_pts_20:
#     ax3.scatter(x, y, color='red', marker='+', s=60, alpha=0.3)
# for x, y in close_Chebyshev_pts_20:
#     ax3.scatter(x, y, color='black', marker='o', s=30, alpha=0.3)
# ax3.scatter(P[0], P[1], color='magenta', marker='D', s=25, alpha = 1)

# shoe=wing plots
plt.show()

# d ii
# The set of points are nearly similar to Eucledian, Manhattan, Minkowski and Chebyshev Distances but they look different for Cosine Distance.
# Euclidean, Manhattan, Minkowski and Chebyshev distance measures are similar and one can see the plots look nearly alike and some of the points overlap as well. Whereas for the Cosine Distance plot, it is different when compared to the other 4 distance metrics. The cosine distance is the dot product of two vectors and the graph somewhat looks like a y=x graph showing us the inclination of all the closest 20 vectors to the data point P (in Magenta color).





##### References #####
"""
1. Euclidean Distance - Lecture3-Data-2022 - page 49
2. Manhattan Distance - Lecture3-Data-2022 - page 52
3. Minkowski Distance - Lecture3-Data-2022 - page 51
4. Chebyshev Distance - https://en.wikipedia.org/wiki/Chebyshev_distance#Definition
5. Cosine Distance - https://en.wikipedia.org/wiki/Cosine_similarity#Definition
"""
##### OUTPUT #####
"""
Mean Point: (61.93852459016394, 15.504098360655737)
close Eucledian points-- [[(62, 15), (63, 15), (63, 17), (63, 14), (63, 14), (60, 15)]]
close Manhattan points-- [[(62, 15), (63, 15), (60, 15), (64, 16), (62, 18), (63, 17)]]
close Minkowski points-- [[(62, 15), (63, 15), (63, 17), (63, 14), (63, 14), (60, 15)]]
close Chebyshev points- [[(62, 15), (63, 15), (63, 17), (63, 14), (63, 14), (60, 14)]]
close Cosine points- [[(56, 14), (60, 15), (64, 16), (84, 21), (75, 19), (67, 17)]]
"""