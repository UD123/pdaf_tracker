#!/usr/bin/env python

'''
Utility functions for PDAF
==================



Usage:

Environemt : 

Install : 


'''

import numpy as np
#import unittest
#import matplotlib.pyplot as plt
#from scipy import interpolate 

 # importing common Use modules 
#import sys 
#%% Logger

import logging 
logger         = logging.getLogger("pdaf")
#formatter   = logging.Formatter('[%(asctime)s.%(msecs)03d] {%(filename)6s:%(lineno)3d} %(levelname)s - %(message)s', datefmt="%M:%S", style="{")
#formatter       = logging.Formatter('[%(asctime)s] - [%(filename)16s:%(lineno)3d] - %(levelname)s - %(message)s')
formatter       = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# file_handler = logging.FileHandler("main_app.log", mode="a", encoding="utf-8")
# file_handler.setLevel("WARNING")
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

#%% 
class TrackState:
    """
    Enumeration class representing the possible states of an object being tracked.
    """
    UNDEFINED       = 0    # No tracking - tracker back to the pool of empty tracks
    FIRST_INIT      = 1    # first tracking init state
    LAST_INIT       = 3    # last track init state
    TRACKING        = 10   # normal tracking everything is ok
    FIRST_COAST     = 20   # first state - track data is lost - waitting to appear
    LAST_COAST      = 25   # last state od tracking without data

#%% Configuration
def config_parameters():
    """
    Initializes different parameters of the algorithm.

    Returns:
        Par: A dictionary containing the algorithm parameters.
    """

    Par                 = {}

    # General management
    Par["ShowOn"]       = True  # Show different information

    # Init data and tracks
    Par["TrajIndex"]    = [12, 11, 8]  # Max 7 trajectories
    Par["PointNum"]     = 9  # More points than trajectories
    Par["NaNDensity"]   = 0.0  # Density of missing points
    Par["Y1Bounds"]     = [0, 1]  # Approximate bounds for Y1 measurements
    Par["Y2Bounds"]     = [0, 1]  # Approximate bounds for Y2 measurements
    Par["TrajNum"]      = len(Par["TrajIndex"])  # Number of trajectories
    Par["Nv"]           = 0.005  # Noise variance
    Par["dT"]           = 1/30  # Time between measurements (seconds)
    Par["Time"]         = 3  # Simulation stopping time (seconds)

    # Kalman filter properties
    Par["StateVar"]     = (0.5)**2  # State variance
    Par["ObserVar"]     = (0.01)**2  # Observation variance

    # Track properties
    Par["TrackNum"]     = 5  # Number of trackers
    Par["ProbDim"]      = 2  # Problem dimensionality (x, x-y, or x-y-z)
    Par["ModelDim"]     = 2  # Constant velocity or constant acceleration
    Par["HistLen"]      = 3  # Number of past states for each tracker
    Par["HistGateLevel"] = 0.1  # History separation value
    Par["LogLikeAlpha"] = 0.3  # Log likelihood forget factor
    Par["GateLevel"]    = 9  # Gating threshold (98.9%)

    # PDAF parameters
    Par["UsePDAF"]      = 0  # Use PDAF mode
    Par["PG"]           = 0.9  # Probability of gating
    Par["PD"]           = 0.8  # Probability of detection

    # Tracker states
    # Par["State_Undefined"] = 0  # Track is free and undefined
    # Par["State_FirstInit"] = 1  # Track in first initialization
    # Par["State_LastInit"]  = 3  # Track in last initialization
    # Par["State_Track"]     = 10  # Track in tracking mode
    # Par["State_FirstCoast"] = 20  # Track in first coast state
    # Par["State_LastCoast"] = 25  # Track in last coast state

    # # Grouping states into action groups
    # Par['State_List_Init']   = list(range(TrackState.FIRST_INIT, TrackState.LAST_INIT + 1))
    # Par['State_List_Coast']  = list(range(TrackState.FIRST_COAST, TrackState.LAST_COAST - 1))
    # Par['State_List_Start']  = [TrackState.TRACKING] + Par['State_List_Init']
    # Par['State_Valid_Show']  = [TrackState.TRACKING] + list(range(TrackState.FIRST_COAST, TrackState.LAST_COAST + 1))

    return Par    

#%% Gaussian distance or probablity
def gaussian_prob(x, m, C, use_log=0):
  """
  Evaluates the multi-variate Gaussian probability density function.

  Args:
    x: Input data matrix (DxN), where N is the number of samples and D is the dimension.
    m: Mean vector (D x 1).
    C: Covariance matrix (DxD).
    use_log: Whether to return the log-probability (0: probability, 1: log-probability, 2: Mahalanobis distance).

  Returns:
    p: Probabilities or log-probabilities or Mahalanobis distances.
  """

  d = len(m)

  if x.shape[0] != d:
    x = x.T

  N = x.shape[1]
  m = m.reshape((-1,1))
  M = np.repeat(m, N, axis=1)

  denom   = (2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(C))
  invC    = np.linalg.inv(C)
  errM    = x - M
  mahal   = np.sum((errM.T @ invC) * errM.T, axis=1) # Chris Bregler's trick

  if use_log == 2:
    p = mahal
  elif use_log == 1:
    p = -0.5 * mahal - np.log(denom)
  else:
    numer = np.exp(-0.5 * mahal)
    p = numer / denom

  return p

