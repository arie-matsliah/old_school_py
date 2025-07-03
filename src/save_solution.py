import numpy as np
import csv

def save_solution(filename, P):
    """
    Saves a permutation matrix P to a CSV solution file.
    """
    # find() in scipy.sparse is equivalent to MATLAB's find
    rows, cols = P.nonzero()
    
    # Convert back to 1-based indexing for the output file
    male_nodes = rows + 1
    female_nodes = cols + 1
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Male Node ID', 'Female Node ID'])
        for i, j in zip(male_nodes, female_nodes):
            writer.writerow([f'm{i}', f'f{j}'])

