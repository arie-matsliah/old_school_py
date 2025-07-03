import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

def read_solution(filename):
    """
    Reads a solution file into a sparse permutation matrix.
    """
    df = pd.read_csv(filename, header=0)
    
    # Extract node IDs
    i = df.iloc[:, 0].str[1:].astype(int).values
    j = df.iloc[:, 1].str[1:].astype(int).values
    
    n = max(i.max(), j.max())
    
    # Create a permutation matrix P where P[i, j] = 1 if male node i maps to female node j
    # Again, assuming 1-based indexing in the file
    return coo_matrix((np.ones(len(i)), (i - 1, j - 1)), shape=(n, n)).tocsr()

