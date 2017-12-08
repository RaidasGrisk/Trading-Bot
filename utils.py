"""
Utils
"""

import numpy as np


def remove_nan_columns(array):
    """
    Deletes columns if at least one value is nan.
    Returns array with filtered columns.
    """
    mask = np.any(np.isnan(array), axis=0)
    return array[:, ~mask]


def prep_data_for_feature_gen(data):
    """Restructure OANDA data to use it for TA-Lib feature generation"""
    inputs = {
        'open': np.array([x['openMid'] for x in data]),
        'high': np.array([x['highMid'] for x in data]),
        'low': np.array([x['lowMid'] for x in data]),
        'close': np.array([x['closeMid'] for x in data]),
        'volume': np.array([float(x['volume']) for x in data])}
    return inputs
