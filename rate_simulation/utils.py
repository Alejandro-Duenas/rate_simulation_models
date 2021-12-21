#---------------------------- Utils Module ------------------------------------
"""
This utils module is a complementary module for the library, which
contains additional functions needed for the correct functioning of
the library.
"""
# ------------------------ Import Libraries -----------------------------------
import numpy as np
import pandas as pd
from typing import Union

#----------------------- Complementary Funcitons ------------------------------
def quant_5(x: np.array)->float:
    """From an array returns the 5th percentile

    Args:
        x (np.array): array of floats

    Returns:
        float: 5th percentile value from the array of floats
    """
    return x.quantile(0.05)

def quant_95(x: np.array)->float:
    """From an array returns the 95th percentile

    Args:
        x (np.array): array of floats

    Returns:
        float: 95th percentile value from the array of floats
    """
    return x.quantile(0.95)

def prop_label(x:Union[float, int])->float:
    """Takes a specific proportion of the input value. This function is
    used in particular to possition the label in the plots of the
    different model classes.

    Args:
        x (Union[float, int]): numerical value

    Returns:
        float: proportion of the input x
    """
    if x<0: return 1.3*x
    else: return 1.05*x

def jump_class(x: Union[float, int])->int:
    """Classifies the type of jump for the Poisson model, so that if the
    condition is met, it goes to the determined below class.

    Args:
        x (Union[float, int]): number of months between jumps

    Returns:
        int: classification number for the lmbda computation.
    """
    if x >= 5: return 5
    elif x == 0: return 1
    else: return x 

def aggregate_random_paths(df: pd.DataFrame)->pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): [description]

    Returns:
        pd.DataFrame: [description]
    """
