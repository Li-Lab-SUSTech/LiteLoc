from numpy import cross, eye, dot
import numpy as np
import csv
# from theano import config
import pandas as pd


def read_csv_array(path):
    """
    Reads a csv_file with columns: [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty,
    z uncertainty, photon uncertainty, x_offset, y_offset]. If the csv_file does not match this format,
    the function will try to find the columns with the following columns: frame, x, y, z, photon... and return them.
    
    Args:
        path (str): path to csv_file
        
    Returns:
        np.ndarray: molecule list
    """

    # molecule_array = pd.read_csv(path, header=None, skiprows=[0]).values

    df = pd.read_csv(path, header=0)

    frame_col = [col for col in df.columns if 'frame' in col]
    x_col = [col for col in df.columns if col == 'x' or col == 'xnm' or col == 'xnano'
             or col == 'x_nm' or col == 'x_nano']
    y_col = [col for col in df.columns if col == 'y' or col == 'ynm' or col == 'ynano'
             or col == 'y_nm' or col == 'y_nano']
    z_col = [col for col in df.columns if col == 'z' or col == 'znm' or col == 'znano'
             or col == 'z_nm' or col == 'z_nano']
    photon_col = [col for col in df.columns if 'photon' in col or 'intensity' in col]

    remaining_cols = [col for col in df.columns if col is not x_col[0] and col is not y_col[0]
                      and col is not z_col[0] and col is not frame_col[0] and col is not photon_col[0]
                      and ('Unnamed' not in col)]

    assert all([frame_col, x_col, y_col, z_col, photon_col]), \
        'Could not find columns with frame,x,y,z,photon in the csv file'

    reordered_cols = [frame_col[0]] + [x_col[0]] + [y_col[0]] + [z_col[0]] + [photon_col[0]] + remaining_cols
    molecule_array = df[reordered_cols].values

    return molecule_array


