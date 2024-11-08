#!/usr/bin/env python

'''

Tracking Object - contains track information, history and kalman filter for a single tracker

==================

Estimates single point position and velocity in 2D and 3D.


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\safety

Install : 
    See README.md


'''

import numpy as np
import unittest
#import logging 
#logger  = logging.getLogger("pdaf")
from utils import logger, gaussian_prob, config_parameters, TrackState
from kalman_filter import KalmanFilter 


# --------------------------------
#%% Main
#%% Additional information for tracking of PDAF
class TrackingObject:
    def __init__(self, params = None, track_id = 1):

        self.params         = config_parameters() if params is None else params 
        self.kf             = KalmanFilter(params["dT"], params["ModelDim"], params["ProbDim"], params["StateVar"], params["ObserVar"])
        self.id             = track_id
        self.state          = TrackState.UNDEFINED
        self.life_time      = 0  # life time indicator

        # data related
        self.data_ind       = []  # list of points id related to the tracker at time t

        # additional book keeping
        self.observ_index   = [0, self.kf.F.shape[0] // 2] # which indx is an observtion variable in the state

        self.history_length = params["HistLen"]
        self.history        = np.zeros((self.kf.F.shape[0], self.history_length))
        self.log_like       = params["GateLevel"]

        self.tprint('Tracker is initialized')

    def init_state(self, data):
        "initializes the position of the tracker"
        self.kf.init_state(data)
        self.state          = TrackState.FIRST_INIT
        self.life_time      = 0
        self.history        = np.zeros((self.kf.F.shape[0], self.history_length))
        return True
    
    def extract_data(self, dataList):
        "extracts the relevant data : data - 2xN - total data at time t"
        # check if any data is associated - otherwise take prediction
        if len(self.data_ind) < 1:
            ydata , stam = self.predict()
        else:
            ydata        = dataList[:,self.data_ind]

        # if not PDAF mode - average all the points in the detected range
        if self.params["UsePDAF"]:
            ydata = np.mean(ydata, axis=1)

        return ydata
    
    def init_velocity(self, ydata):
        "initializes the velocity of the tracker"

        if self.state == TrackState.FIRST_INIT:
            y       = np.mean(ydata, axis=1)
            self.kf.init_velocity(y)

        return True    
    
    def check_valid(self):
        "check if the states are defined"
        return self.state != TrackState.UNDEFINED
    
    def predict(self):
        "predicting the state"
        # Prediction of the next state
        ypred, Spred = self.kf.predict()
        return ypred, Spred

    def association_distance(self, data):
        " compuites data point association distance"
        " Input : data 2 x N points in 2D space"
        " Ouput : dist 1 x N distance/probaility matrix"

        # Prediction of the next state
        ypred, Spred = self.kf.predict()

        # compute distance
        dist         = gaussian_prob(data, ypred, Spred, 2)

        return dist
    
    def update_state(self):
        "computes the next state according the data association and the current state"
        # Check for associated data

        no_data_associated  = len(self.data_ind) < 1
        state_current       = self.state
        state_next          = self.state

        if no_data_associated:
            # No data associated
            if state_current == TrackState.UNDEFINED:
                raise ValueError("Undefined track")
            elif TrackState.FIRST_INIT <= state_current and state_current <= TrackState.LAST_INIT:
                # Initialization states - no data - delete the tracker
                state_next = TrackState.UNDEFINED  # Reset initialization
            elif TrackState.FIRST_COAST <= state_current and state_current < TrackState.LAST_COAST: 
                # Coast mode states : all coast states except the last one
                state_next = state_current + 1  # Next coast mode state
            elif state_current == TrackState.LAST_COAST:
                # Final coast mode state
                state_next = TrackState.UNDEFINED  # Reset track
                self.tprint('Deleted')
            else:
                # Track state
                state_next = TrackState.FIRST_COAST  # First coast mode state
                self.tprint('Coast Mode')
        else:
            # Data associated
            if state_current == TrackState.UNDEFINED:
                raise ValueError("Undefined track")
            elif TrackState.FIRST_INIT <= state_current and state_current < TrackState.LAST_INIT:
                # Initialization states
                state_next = state_current + 1  # Next initialization state
            elif state_current == TrackState.LAST_INIT:
                # Last initialization state
                state_next = TrackState.TRACKING  # Next track state
                self.tprint('From Init to Tracking')
            elif TrackState.FIRST_COAST <= state_current and state_current <= TrackState.LAST_COAST: 
                # Coast mode states
                state_next = TrackState.TRACKING  # Return to track state
                self.tprint('From Coast to Tracking')
            else:
                # Track state
                state_next = TrackState.TRACKING  # Stay in track state

        self.state = state_next
        return 

    def update(self, y):
        "predicting the state and computing the update"
        PG          = self.params["PG"]
        PD          = self.params["PD"]    
        GateLevel   = self.params["GateLevel"]

        if self.params["UsePDAF"]:
            ret = self.kf.update_pdaf(self, y, PG, PD, GateLevel) 
        else:
            ret = self.kf.update(self, y)

        return ret  

    def update_statistics(self):
        "manage some debug info"
        
        # Update track history
        self.history[:, 1:]    = self.history[:, :-1]
        self.history[:, 0:1]   = self.kf.x  # 0 is not working !!!!

        # Update track lifetime
        self.life_time += 1

        return True

    def compute_likelihood(self):
        "likelihood tracking"
        alpha       = self.params["LogLikeAlpha"]
        PG          = self.params["PG"]
        PD          = self.params["PD"]
            
        # Update log-likelihood
        if TrackState.FIRST_COAST <= self.state and self.state <= TrackState.LAST_COAST:
            log_like = self.params["GateLevel"]
        else:
            log_like = self.kf.loglike
            
        self.log_like = (1 - alpha) * self.log_like + alpha * log_like
        return True 
    
    def get_show_info(self):
        "extract information useful for show"
        
        ypred, Spred = self.kf.predict()
        yhist        = self.history[self.observ_index,:] 
        
        return ypred, Spred, yhist

    def finish(self):
        # Close down 
        self.tprint('Finished')

    def tprint(self, txt = '', level = 'I'):
        txt = '%03d : %s' %(self.id, txt)
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
class TestTrackingObject(unittest.TestCase):

    def test_create(self):
        "testing creation"
        cfg     = config_parameters()
        p       = TrackingObject(cfg, 1)
        data    = np.eye(2)
        retp    = p.init_state(data[:,0])
        prob    = p.association_distance(data)
        ret     = p.check_valid()
        p.finish()
        self.assertTrue(not ret) 

          

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestTrackingObject("test_create")) # ok

    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()


