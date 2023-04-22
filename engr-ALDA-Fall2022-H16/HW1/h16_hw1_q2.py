# NumPy Reference Documentation - https://numpy.org/doc/stable/reference/index.html
import numpy as np
import numpy.matlib as mtl
import math
np.random.seed(2022)

# (a) Generate a 5*5 identity matrix A.
A = np.identity(5)
print("1) A 5*5 identity matrix A--\n", A)


# (b) Change all elements in the 5th column of A to 5.
A[:,4] = 5
print("\n2) Changing elements of 5th column of A to 5--\n", A)


# (c) Sum of all elements in the matrix (use ONE “for/while loop”).
print("\n3) Sum of all elements in the matrix--", np.sum(A))


# (d) Transpose the matrix A.
A_T = A.transpose()
print("\n4) Transpose the matrix A--\n", A_T)


# (e) Calculate the sum of the 5th row, the sum of the diagonal and the sum of the 1st column in matrix A, respectively.
print("\n5) Sum of 5th row, sum of diagonal, sum of 1st column in matrix A (Transposed)--")
print(np.sum(A_T[4, :]), np.sum(np.diagonal(A_T)), np.sum(A_T[:, 0]))


# (f) Generate a 5*5 matrix B following standard normal distribution.
B = mtl.randn(1, 5, 5)
print("\n6) 5*5 matrix B following standard normal distribution--\n", B)


# (g) From A and B, using matrix operations to get a new 2*5 matrix C such that,the first row of C is equal to the 1st row of B minus the 1st row of A, the second row of C is equal to the sum of the 5th row of A and the 5th row of B.
C = np.concatenate([np.subtract(A[0,:], B[0,:]), np.add(A[4,:], B[4,:])], axis=0)
print("\n7) A new 2*5 matrix C--\n", C)


# (h) From C, using ONE matrix operation to get a new matrix D such that,the first column of D is equal to the first column of C, the second column of D is equal to the second column of C times 2, the third column of D is equal to the third column of C times 3, and so on.
D = np.concatenate([C[:,0], C[:,1]*2, C[:,2]*3, C[:,3]*4, C[:,4]*5], axis=1)
print("\n8) A new matrix D--\n", D)


# (i) X = [1,1,1,2]T, Y = [0,3,6,9]T, Z = [4,3,2,1]T. Compute the co- variance matrix of X, Y, and Z. Then compute the Pearson correlation coefficients between X and Y.
X = [1,1,1,2]
Y = [0,3,6,9]
Z = [4,3,2,1]
print("\n9)\n i. Covariance matrix of X, Y, Z--\n", np.cov(np.stack((X, Y, Z))))
print("\n ii. Pearson Correlation Coefficients between X and Y--\n", np.corrcoef(X, Y))


# (j) Verify the equation: x2 ̄ = (x ̄2+σ2(x)) using x = [23, 19, 21, 22, 21, 23, 23, 20]T when (python library math is allowed):
x = [23, 19, 21, 22, 21, 23, 23, 20]
xn = len(x)
xmean = np.mean(x)
x_squared_mean = sum([i ** 2 for i in x]) / xn
xmean_squared = xmean ** 2
x_xmean_squared = [(i - xmean) ** 2 for i in x]

# i. σ(x) is the population standard deviation. Show your work.
 # population std = sqrt( sum ( x  - x_mean ) square / ( n ) )
pstd = math.sqrt( sum(x_xmean_squared) / xn )
pstd_squared = pstd ** 2
print("\n10)\n i. LHS = mean of x^2 = ", x_squared_mean)
print("    RHS = x_mean^2 + std^2(x) = ", xmean_squared + pstd_squared)
truth_flag = "LHS = RHS" if (x_squared_mean == (xmean_squared + pstd_squared)) else "there is a difference of " + str(xmean_squared + pstd_squared - x_squared_mean) + "between LHS and RHS."
print("    Hence, with Population Standard Deviation,", truth_flag)

# ii. σ(x) is the sample standard deviation. Show your work.
 # sample std = sqrt( sum ( x  - x_mean ) square / ( n - 1 ) )
sstd = math.sqrt( sum(x_xmean_squared) / (xn - 1) )
sstd_squared = sstd ** 2
print("\n ii. LHS = mean of x^2 = ", x_squared_mean)
print("     RHS = x_mean^2 + std^2(x) = ", xmean_squared + sstd_squared)
truth_flag = "LHS = RHS" if (x_squared_mean == (xmean_squared + sstd_squared)) else "there is a difference of " + str(xmean_squared + sstd_squared - x_squared_mean) + " between LHS and RHS."
print("     Hence, with Sample Standard Deviation,", truth_flag)




##### REFERENCES #####
"""
1. Standard Deviation - https://en.wikipedia.org/wiki/Standard_deviation
"""

##### OUTPUT #####
"""
1) A 5*5 identity matrix A--
 [[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]

2) Changing elements of 5th column of A to 5--
 [[1. 0. 0. 0. 5.]
 [0. 1. 0. 0. 5.]
 [0. 0. 1. 0. 5.]
 [0. 0. 0. 1. 5.]
 [0. 0. 0. 0. 5.]]

3) Sum of all elements in the matrix-- 29.0

4) Transpose the matrix A--
 [[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [5. 5. 5. 5. 5.]]

5) Sum of 5th row, sum of diagonal, sum of 1st column in matrix A (Transposed)--
25.0 9.0 6.0

6) 5*5 matrix B following standard normal distribution--
 [[-5.27899086e-04 -2.74901425e-01 -1.39285562e-01  1.98468616e+00
   2.82109326e-01]
 [ 7.60808658e-01  3.00981606e-01  5.40297269e-01  3.73497287e-01
   3.77813394e-01]
 [-9.02131926e-02 -2.30594327e+00  1.14276002e+00 -1.53565429e+00
  -8.63752018e-01]
 [ 1.01654494e+00  1.03396388e+00 -8.24492228e-01  1.89048564e-02
  -3.83343556e-01]
 [-3.04185475e-01  9.97291506e-01 -1.27273841e-01 -1.47588590e+00
  -1.94090633e+00]]

7) A new 2*5 matrix C--
 [[ 1.0005279   0.27490142  0.13928556 -1.98468616  4.71789067]
 [-0.30418547  0.99729151 -0.12727384 -1.4758859   3.05909367]]

8) A new matrix D--
 [[ 1.0005279   0.54980285  0.41785668 -7.93874463 23.58945337]
 [-0.30418547  1.99458301 -0.38182152 -5.90354361 15.29546836]]

9)
 i. Covariance matrix of X, Y, Z--
 [[ 0.25        1.5        -0.5       ]
 [ 1.5        15.         -5.        ]
 [-0.5        -5.          1.66666667]]

 ii. Pearson Correlation Coefficients between X and Y--
 [[1.         0.77459667]
 [0.77459667 1.        ]]

10)
 i. LHS = mean of x^2 =  464.25
    RHS = x_mean^2 + std^2(x) =  464.25
    Hence, with Population Standard Deviation, LHS = RHS

 ii. LHS = mean of x^2 =  464.25
     RHS = x_mean^2 + std^2(x) =  464.5357142857143
     Hence, with Sample Standard Deviation, there is a difference of 0.2857142857142776 between LHS and RHS.
"""