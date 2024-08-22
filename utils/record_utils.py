import numpy as np
import csv
import pandas as pd
import torch
import os

from utils.help_utils import *


def read_csv(path, flip_z=False, z_fac=1, pix_scale=[1, 1], drift_txt=None):
    """Reads a csv_file with leading columns: 'localization', 'frame', 'x', 'y', 'z'

    Parameters
    ----------
    flip_z : bool
        If True flips the z variable
    z_fac: float
        Multiplies z variable with the given factor to correct for eventual aberrations
    pix_scale: list of two ints
        Multiplies x and y locations with the given factors
    drift_txt : str
        Reads a drift corredtion txt file and applies drift correction for x and y locations

    Returns
    -------
    preds :list
        List of localizations with x,y,z locations given in nano meter
    """
    preds = pd.read_csv(path, header=None, skiprows=[0]).values

    if drift_txt is not None:
        drift_data = pd.read_csv(drift_txt, sep='	', header=None, skiprows=0).values

        for p in preds:
            p[2] = float(p[2]) - 100 * drift_data[np.clip(int(p[1]) - 1, 0, len(drift_data) - 1), 1]
            p[3] = float(p[3]) - 100 * drift_data[np.clip(int(p[1]) - 1, 0, len(drift_data) - 1), 2]

    preds[:, 2] = preds[:, 2] * pix_scale[0] + 0
    preds[:, 3] = preds[:, 3] * pix_scale[1] + 0
    preds[:, 4] = preds[:, 4] * z_fac + 0
    if flip_z:
        preds[:, 4] = -preds[:, 4] + 0

    return preds


def check_csv(name):
    ind = 0
    if os.path.exists(name):
        with open(name) as csvfile:
            mLines = csvfile.readlines()

        targetLine = mLines[-1]
        ind = targetLine.split(',')[0]
    else:
        with open(name, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            csvwriter.writerow(['frame', 'xnano', 'ynano', 'znano', 'intensity','pnms'])
    return 0



def write_csv(pred_list, name):
    """Writes a csv_file with columns: 'localizatioh', 'frame', 'x', 'y', 'z', 'intensity','x_sig','y_sig','z_sig'

    Parameters
    ----------
    pred_list : list
        List of localizations
    name: str
        File name
    """
    with open(name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for p in pred_list:
            csvwriter.writerow([repr(f) for f in p])
    # if write_gt:
    #     with open(name, 'w', newline='') as csvfile:
    #         csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #         csvwriter.writerow(['x_gt', 'y_gt', 'z_gt', 'intensity_gt', 'x_pred', 'y_pred', 'z_pred',
    #                             'intensity_pred', 'nms_p', 'x_sig', 'y_sig', 'z_sig'])
    #         for p in pred_list:
    #             csvwriter.writerow([repr(f) for f in p])
    # else:
    #     if os.path.exists(name):
    #         with open(name, 'a', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #             for p in pred_list:
    #                 csvwriter.writerow([repr(f) for f in p])
    #     else:
    #         with open(name, 'w', newline='') as csvfile:
    #             csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    #             csvwriter.writerow(['frame', 'xnano', 'ynano', 'znano', 'intensity'])
    #             for p in pred_list:
    #                 csvwriter.writerow([repr(f) for f in p])





