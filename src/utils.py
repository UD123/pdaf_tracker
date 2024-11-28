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

# Check if a logger named 'my_logger' is already defined
if not logger.handlers:
    # Logger is not configured, so configure it

    #formatter   = logging.Formatter('[%(asctime)s.%(msecs)03d] {%(filename)6s:%(lineno)3d} %(levelname)s - %(message)s', datefmt="%M:%S", style="{")
    formatter       = logging.Formatter('[%(asctime)s] - [%(filename)18s:%(lineno)3d] - %(levelname)5s - %(message)s')
    #formatter       = logging.Formatter('[%(asctime)s] - %(levelname)s - %(message)s')
    logger.setLevel("DEBUG")
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel("DEBUG")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("pdaf_tracking.log", mode="a", encoding="utf-8")
    file_handler.setLevel("WARNING")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    Par["PointNum"]     = 2  # More points than trajectories
    Par["NaNDensity"]   = 0.0  # Density of missing points
    Par["Y1Bounds"]     = [0, 1]  # range/ bounds for Y1 measurements
    Par["Y2Bounds"]     = [0, 1]  # range / bounds for Y2 measurements
    Par["TrajNum"]      = len(Par["TrajIndex"])  # Number of trajectories
    Par["Nv"]           = 0.005  # Noise variance : relative to 0-1 range
    Par["dT"]           = 1/30  # Time between measurements (seconds)
    Par["Time"]         = 3  # Simulation stopping time (seconds)

    # Kalman filter properties
    Par["StateVar"]     = (0.5)**2  # State variance
    Par["ObserVar"]     = (0.03)**2  # Observation variance

    # Track properties
    Par["TrackNum"]     = 1  # Number of trackers
    Par["ProbDim"]      = 2  # Problem dimensionality (x, x-y, or x-y-z)
    Par["ModelDim"]     = 2  # Constant velocity or constant acceleration
    Par["HistLen"]      = 10  # Number of past states for each tracker
    Par["HistGateLevel"] = 0.1  # History separation value
    Par["LogLikeAlpha"] = 0.3  # Log likelihood forget factor
    Par["GateLevel"]    = (2.3)**2  # for mahal distance - err*invCov*err 
    #                               # for probabilitu=ies and PDAF  - Gating threshold (98.9%)

    # PDAF parameters
    Par["UsePDAF"]      = 0  # Use PDAF mode
    Par["PG"]           = 0.9  # Probability of gating
    Par["PD"]           = 0.8  # Probability of detection

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

