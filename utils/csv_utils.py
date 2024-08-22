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


def write_csv_array(input_array, filename, write_mode='write localizations'):
    """
    Writes a csv_file with different column orders depending on the input.
        [frame, x, y, z, photon, integrated prob, x uncertainty, y uncertainty,
         z uncertainty, photon uncertainty, x_offset, y_offset]
    
    Args:
        input_array (np.ndarray): molecule array that need to be written
        filename (str): path to csv_file
        write_mode (str):
            1. 'write paired localizations': write paired ground truth and predictions from the
                ailoc.common.assess.pair_localizations function, the format is
                ['frame', 'x_gt', 'y_gt', 'z_gt', 'photon_gt', 'x_pred', 'y_pred', 'z_pred', 'photon_pred'];

            2. 'write localizations': write predicted molecule list, the format is
                ['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig', 'photon_sig',
                'xo', 'yo'];

            3. 'append localizations': append to existing file using the format in 2;

            4. 'write rescaled localizations': write predicted molecule list with rescaled coordinates, the format is
                ['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig', 'photon_sig',
                'xo', 'yo', 'xo_rescale', 'yo_rescale', 'xnm_rescale', 'ynm_rescale'];
    """

    if write_mode == 'write paired localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'x_gt', 'y_gt', 'z_gt', 'photon_gt', 'x_pred', 'y_pred', 'z_pred',
                                'photon_pred'])
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    elif write_mode == 'write localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig',
                                'photon_sig', 'xoffset', 'yoffset'])
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    elif write_mode == 'append localizations':
        with open(filename, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    elif write_mode == 'write rescaled localizations':
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnm', 'ynm', 'znm', 'photon', 'prob', 'x_sig', 'y_sig', 'z_sig',
                                'photon_sig', 'xo', 'yo', 'xo_rescale', 'yo_rescale', 'xnm_rescale',
                                'ynm_rescale'])
            for row in input_array:
                csvwriter.writerow([repr(element) for element in row])
    else:
        raise ValueError('write_mode must be "write paired localizations", "write localizations", '
                         '"append localizations", or "write rescaled localizations"')
