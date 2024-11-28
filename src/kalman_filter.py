#!/usr/bin/env python

'''

Kalman Filter - Used in a Single Tracker

==================

Estimates point position and velocity in 2D and 3D.


Usage:

Install : 
    See README.md


'''

import numpy as np
import unittest
from utils import logger

# --------------------------------
#%% Main
class KalmanFilter:

    def __init__(self, dT = 0.1, ModelDim = 2, ProbDim = 2, StateVar = 0.01, ObserVar=0.01):

        # params
        F, H, Q, R, observ_ind, initx, initP = self.init_kalman_filter(dT, ModelDim, ProbDim, StateVar, ObserVar=ObserVar)

        # transition and noise matrices
        self.F              = F  # state transition
        self.H              = H  # observation matrix
        self.Q              = Q  # state noise
        self.R              = R  # observation noise

        # state variables
        self.x              = initx   
        self.P              = initP    

        # performance indicator
        self.loglike        = 0        

        # additional
        self.observ_index   = observ_ind # which indx is an observtion variable in the state
        self.initP          = initP      # uncertainity at the first time
        self.dT             = dT         # time difference between samples

        # information display
        self.debug_level    = 0

        logger.debug('Kalman Filter is initialized')

    def init_kalman_filter(self, dT, ModelDim, ProbDim, StateVar, ObserVar=0.01):
        """
        Initializes Kalman filter matrices for 2D and 3D (TBD) cases.

        Args:
            dT: Time step.
            ModelDim: Model dimension (2 for constant velocity, 3 for constant acceleration).
            ProbDim: Problem dimension (2D, 3D).
            StateVar: State variance.
            ObserVar: Observation variance.

        Returns:
            F: State transition matrix.
            H: Observation matrix.
            Q: Process noise covariance.
            R: Measurement noise covariance.
            ObservInd: Indices of observable variables.
            initx: Initial state vector.
            initP: Initial state covariance matrix.
        """

        # Initialize matrices based on model and problem dimensions
        if ModelDim == 2:
            Ftmp = np.array([[1, dT], [0, 1]])
            Htmp = np.array([[1, 0]])
            Qtmp = np.array([[dT**4/4, dT**3/2], [dT**3/2, dT**2]]) * StateVar
            Rtmp = np.eye(1) * ObserVar
            xtmp = np.zeros((2, 1))
            Ptmp = Qtmp * 10

        elif ModelDim == 3:
            Ftmp = np.array([[1, dT, dT**2/2], [0, 1, dT], [0, 0, 1]])
            Htmp = np.array([[1, 0, 0]])
            Qtmp = np.array([[dT**4/20, dT**3/8, dT**2/6],
                            [dT**3/8, dT**2/3, dT/2],
                            [dT**2/6, dT/2, 1]]) * dT * StateVar
            Rtmp = np.eye(1) * ObserVar
            xtmp = np.zeros((3, 1))
            Ptmp = Qtmp * 100

        else:
            raise ValueError("Unsupported model dimension.")

        if ProbDim == 2:
            F       = np.block([[Ftmp, np.zeros_like(Ftmp)], [np.zeros_like(Ftmp), Ftmp]])
            H       = np.block([[Htmp, np.zeros_like(Htmp)], [np.zeros_like(Htmp), Htmp]])
            Q       = np.block([[Qtmp, np.zeros_like(Qtmp)], [np.zeros_like(Qtmp), Qtmp]])
            R       = np.block([[Rtmp, np.zeros_like(Rtmp)], [np.zeros_like(Rtmp), Rtmp]])
            initP   = np.block([[Ptmp, np.zeros_like(Ptmp)], [np.zeros_like(Ptmp), Ptmp]])
            initx   = np.vstack([xtmp, xtmp])
            ObservInd = np.array([0, F.shape[0] // 2], dtype = np.int32)
        else:
            raise ValueError("Unsupported problem dimension.")

        logger.debug(f'Kalamn initialized with state variance {StateVar} and observation variance {ObserVar}')
        return F, H, Q, R, ObservInd, initx, initP    

    def init_state(self, data):
        "assigns a new position and state variance" 
        dim_num = len(self.observ_index)
        data    = data.reshape((-1,1))
        if dim_num != data.shape[0]:
            raise ValueError('input data must be 2 dimensional')
            #logger.debug('input data must be 2 dimensional','E')
        
        # state position
        self.x[self.observ_index + 0] = data # position in 2D/3D
        self.x[self.observ_index + 1] = 0    # velocity 0
        self.P                        = self.initP

        return True
    
    def init_velocity(self, y):
        "after the first detection - start velocity. y - Dx1 data"
        xpred = self.F @ self.x
        ypred = self.H @ xpred
        veloc   = (y.reshape((-1,1)) - ypred) / self.dT
        self.x[self.observ_index + 1] = veloc

    def predict(self):
        " extracts prediction : measurment and covariance "

        # Prediction
        xpred = self.F @ self.x
        Ppred = self.F @ self.P @ self.F.T + self.Q
        ypred = self.H @ xpred
        Spred = self.H @ Ppred @ self.H.T + self.R   

        return ypred, Spred
    
    def update(self, y):
        "predicting the state and computing the update"
        # make input a column vector
        y               = y.reshape((-1,1))

        # Prediction of the next state - can be saved from previous step
        xpred           = self.F @ self.x
        Ppred           = self.F @ self.P @ self.F.T + self.Q
        ypred           = self.H @ xpred
        Spred           = self.H @ Ppred @ self.H.T + self.R
        Sinv            = np.linalg.inv(Spred)

        # Kalman filter core
        e               = y - np.tile(ypred, y.shape[1])  # Error (innovation) for each sample
        K               = Ppred @ self.H.T @ Sinv         # Kalman gain
        Pc              = (np.eye(self.F.shape[0]) - K @ self.H) @ Ppred   

        etot            = e
        Pnew            = Pc
        xnew            = xpred + K @ etot
        
        # Update track state
        self.x          = xnew
        self.P          = Pnew

        # performance indicator
        self.loglike    = (etot.T @ Sinv @ etot).squeeze()

        return True
    
    def update_pdaf(self, y, PG, PD, GateLevel):
        "PDAF update"
        # make input a column vector
        assert y.shape[0] == 2, 'y data vector must be 2 x N array'
        point_num       = y.shape[1]

        # Prediction of the next state - can be saved from previous step
        xpred           = self.F @ self.x
        Ppred           = self.F @ self.P @ self.F.T + self.Q
        ypred           = self.H @ xpred
        Spred           = self.H @ Ppred @ self.H.T + self.R
        Sinv            = np.linalg.inv(Spred)  

        # Kalman filter core
        
        e               = y - np.tile(ypred, point_num)  # Error (innovation) for each sample
        K               = Ppred @ self.H.T @ Sinv         # Kalman gain
        Pc              = (np.eye(self.F.shape[0]) - K @ self.H) @ Ppred                
                
        # Compute association probabilities
        loglik          = np.sum(np.dot(e.T, Sinv) * e, axis=1)
        betta           = np.zeros(point_num + 1)
        betta[:point_num] = np.exp(-0.5 * loglik)
        betta[point_num] = (1 - PG * PD) / PD * 2 * point_num / GateLevel * np.sqrt(np.linalg.det(Spred))
        betta           = betta/np.sum(betta)
        
        # Update
        etot            = e @ betta[:point_num].T
        Pgag            = K @ ((e * betta[:point_num, None]) @ e.T - etot @ etot.T) @ W.T
        Pnew            = betta[point_num] * Ppred + (1 - betta[point_num]) * Pc + Pgag

        Pnew            = Pc
        xnew            = xpred + K @ etot
        
        # Update track state
        self.x          = xnew
        self.P          = Pnew  

        # performance indicator
        self.loglike    = etot.T @ Sinv @ etot        

        return True      


    def finish(self):
        # Close down 
        logger.debug('Finished')


    def tprint(self, txt = '', level = 'I'):
        if self.debug_level < 1:  # control the print
            return

        txt = 'KLM : ' + txt
        if level == "I":
            logger.info(txt)
        elif level == "W":
            logger.warning(txt)
        elif level == "E":
            logger.error(txt)
        else:
            logger.info(txt)
 
# --------------------------------
#%% Tests
class TestKalmanFilter(unittest.TestCase):

    def test_create(self):
        "create and some functionality"
        p       = KalmanFilter()
        data    = np.eye(2)
        ret     = p.init_state(data[:,0])
        y,S     = p.predict()
        p.finish()
        self.assertTrue(ret) 

    def test_update(self):
        "create and some functionality"
        p       = KalmanFilter()
        data    = np.eye(2)
        ret     = p.init_state(data[:,0])
        ret     = p.init_velocity(data[:,1])
        ret     = p.update(data[:,1])
        p.finish()
        self.assertTrue(ret)         

          

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestKalmanFilter("test_create")) # ok
    suite.addTest(TestKalmanFilter("test_update")) # ok
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()


