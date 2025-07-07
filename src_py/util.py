import csv
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

_DEBUG_ = False

start_time = time.time()  # Record the start time

def log(message):
    elapsed = time.time() - start_time
    print(f"{message}    [{elapsed:.2f}s]")

def dbg(message):
    if _DEBUG_:
        log(message)


def read_connectome(filename):
    df = pd.read_csv(filename, header=0)
    # Extract node IDs, assuming format 'm123' or 'f123'
    i = df.iloc[:, 0].str[1:].astype(int).values
    j = df.iloc[:, 1].str[1:].astype(int).values
    w = df.iloc[:, 2].astype(float).values
    n = max(i.max(), j.max())
    return coo_matrix((w, (i - 1, j - 1)), shape=(n, n)).tocsr()


def read_solution(filename):
    df = pd.read_csv(filename, header=0)
    i = df.iloc[:, 0].str[1:].astype(int).values
    j = df.iloc[:, 1].str[1:].astype(int).values
    n = max(i.max(), j.max())
    # Create a permutation matrix P where P[i, j] = 1 if male node i maps to female node j
    return coo_matrix((np.ones(len(i)), (i - 1, j - 1)), shape=(n, n)).tocsr()


def save_solution(filename, P):
    rows, cols = P.nonzero()
    male_nodes, female_nodes = rows + 1, cols + 1
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Male Node ID', 'Female Node ID'])
        for i, j in zip(male_nodes, female_nodes):
            writer.writerow([f'm{i}', f'f{j}'])
