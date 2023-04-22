import pandas as pd
import numpy as np
import math
from tabulate import tabulate

arr = {'A':[5,-3], 'B':[-1,5], 'C':[2,1], 'D':[-4,-4], 'E':[4,5], 'F':[1,-2], 'G':[-3,6], 'H':[4,4], 'I':[3,-1], 'J':[3,-2], 'K':[3,-3]}
G = [-3,6]
F = [1,-2]
A = [5,-3]
arrG = {}
arrF = {}
arrA = {}
for i in range(4):
    CG = []
    CF = []
    CA = []
    print(f"\n\n***** {i+1} *****")
    for j in arr:
        arrG[j] = math.dist(arr[j],G)
        arrF[j] = math.dist(arr[j],F)
        arrA[j] = math.dist(arr[j],A)

        if arrA[j] < arrF[j] and arrA[j] < arrG[j]:
            CA.append(j)
        elif arrF[j] < arrG[j]:
            CF.append(j)
        else:
            CG.append(j)

    df = pd.DataFrame([arrG,arrF,arrA], index=['G','F','A'])
    print(tabulate(df, headers = 'keys', tablefmt = "fancy_grid"))

    G = [np.mean([arr[k][0] for k in CG]), np.mean([arr[k][1] for k in CG])]
    F = [np.mean([arr[k][0] for k in CF]), np.mean([arr[k][1] for k in CF])]
    A = [np.mean([arr[k][0] for k in CA]), np.mean([arr[k][1] for k in CA])]
    print()
    print(G, CG)
    print(F, CF)
    print(A, CA)
