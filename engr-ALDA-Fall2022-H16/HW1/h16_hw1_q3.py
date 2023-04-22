# pandas Documentation - https://pandas.pydata.org/docs/
# numpy Reference Documentation - https://numpy.org/doc/stable/reference/index.html
# matplotlib Reference Documentation - https://matplotlib.org/2.0.2/index.html
# statsmodel Reference Documentation - https://www.statsmodels.org/stable/index.html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab as py

filename = input("Enter the path to the filename which contains the data: ")

#3.a
df = pd.read_csv(filename)
output = df[["area","perimeter","length of kernel","width of kernel"]].describe().loc[["mean","std","25%","50%","75%"]]
print ("############################################\n")
print (f"Values for mean, standard deviation, 25%, 50% and 75% for the data is: \n{output}")

range = df[["area","perimeter","length of kernel","width of kernel"]].max()-df[["area","perimeter","length of kernel","width of kernel"]].min()
print ("\n")
print ("############################################\n")
print (f"Range for the defined attributes is: \n{range}")

median = df.describe().median()
print ("\n")
print ("############################################\n")
print (f"Median for the defined attributes is: \n{median}")

#3.b
df = pd.read_csv(filename)
data_plot = df[["length of kernel","width of kernel","class"]]

data_plot.head()

#fig = plt.figure(figsize = (10, 7))
#ax = fig.add_axes([0, 0, 1, 1])

data_plot.boxplot(column=['length of kernel','width of kernel'], by=['class'])

plt.show()


#3.c
df = pd.read_csv(filename)
df.hist(column=['compactness','asymmetry coefficient'],bins=16)

plt.show()
#3.d
df = pd.read_csv(filename)
#att is a variable to store requried attributes
att=['area','length of kernel','width of kernel','compactness']
#converting the data type of below attributes  to numeric
df['area'] = pd.to_numeric(df['area'])
df['length of kernel'] = pd.to_numeric(df['length of kernel'])
df['width of kernel'] = pd.to_numeric(df['width of kernel'])
df['compactness'] = pd.to_numeric(df['compactness'])

pd.plotting.scatter_matrix(df[att],diagonal='kde',figsize=(10,10),c=df['class'])
plt.show()

#3.e

df = pd.read_csv(filename)
#adjust the figure size by using figsize
plt.figure(figsize=(10,8))
ax=plt.axes(projection='3d')
fg=ax.scatter3D(df['length of kernel'],df['width of kernel'],df['area'],c=df['class'])
#label x-axis
ax.set_xlabel('length of kernel')
#label y-axix
ax.set_ylabel('width of kernel')
#label z-axis
ax.set_zlabel('area')

#3.f
df = pd.read_csv(filename)
sm.qqplot(df[["length of kernel"]], line ='45')
sm.qqplot(df[["compactness"]], line ='45')
py.show()





##### OUTPUT #####
"""
Enter the path to the filename which contains the data: seeds_dataset.csv
############################################

Values for mean, standard deviation, 25%, 50% and 75% for the data is: 
           area  perimeter  length of kernel  width of kernel
mean  14.847524  14.559286          5.628533         3.258605
std    2.909699   1.305959          0.443063         0.377714
25%   12.270000  13.450000          5.262250         2.944000
50%   14.355000  14.320000          5.523500         3.237000
75%   17.305000  15.715000          5.979750         3.561750


############################################

Range for the defined attributes is: 
area                10.590
perimeter            4.840
length of kernel     1.776
width of kernel      1.403
dtype: float64


############################################

Median for the defined attributes is: 
area                       14.601262
perimeter                  14.439643
compactness                 0.872224
length of kernel            5.576017
width of kernel             3.247802
asymmetry coefficient       3.649600
length of kernel groove     5.315536
class                       2.000000
dtype: float64
"""