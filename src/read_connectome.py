import pandas as pd
from scipy.sparse import coo_matrix

def read_connectome(filename):
    """
    Reads a connectome from a CSV file into a sparse matrix.
    """
    df = pd.read_csv(filename, header=0)
    
    # Extract node IDs, assuming format 'm123' or 'f123'
    i = df.iloc[:, 0].str[1:].astype(int).values
    j = df.iloc[:, 1].str[1:].astype(int).values
    w = df.iloc[:, 2].astype(float).values
    
    n = max(i.max(), j.max())
    
    # Use COO matrix format, which is efficient for construction
    # MATLAB sparse is 1-based, Python is 0-based. The data seems 1-based.
    # We will assume the node IDs are 1-based and convert to 0-based.
    return coo_matrix((w, (i - 1, j - 1)), shape=(n, n)).tocsr()

