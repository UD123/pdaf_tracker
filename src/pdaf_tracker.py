#!/usr/bin/env python

'''

PDAF Point Tracker

==================

Using IPDAF Filters to predict, track and estimate position of multiple points.
It can deal with occlusion, tracking loss, intersection and different point motion profiles.


Usage:

Environemt : 
    C:\\Users\\udubin\\Documents\\Envs\\safety

Install : 
    See README.md


'''

import numpy as np
import time
import unittest

from data_generator import DataGenerator
from track_object   import TrackingObject
from data_display   import DataDisplay
from utils          import logger, config_parameters


# --------------------------------
#%% Main
class PDAF:

    def __init__(self):

        # params
        self.params          = self.init_parameters()
        self.time_counter    = 0

        logger.info(f'Created')

    def init_parameters(self):
        """
        Initializes different parameters of the algorithm.

        Returns:
            Par: A dictionary containing the algorithm parameters.
        """
        par                 = config_parameters()

        track_num           = par['TrackNum']
        logger.info(f'Track number : {track_num}')
        return par     

    def init_tracks(self, par = None):
        """
        Initializes the Kalman filter matrix and other track-related data.

        Args:
            Par: A dictionary containing parameters.

        Returns:
            TrackList: A list of track objects.
        """
        par         = self.params if par is None else par

        track_num   = par["TrackNum"]


        # Initialize the track list
        trackList = []
        for i in range(track_num):

            track       = TrackingObject(par, i+1)
            trackList.append(track)

        return trackList
    
    def track_association(self, trackList, dataList, par):
        """
        Associates data points with tracks.
        Trackers in the tracking state are assigned with data points.
        Trackers in undefined state assigned with the rest of the points.

        Args:
            trackList: Kalman structure list and more
            dataList:  2 x DataPointsNum contains the relevant data for time t
            Par: Dictionary containing parameters.

        Returns:
            trackList: Updated list of track objects.
        """

        GateLevel   = par["GateLevel"]
        TrackNum    = par["TrackNum"]

        # deal with no data provided - empty data list
        if len(dataList) < 1:
            logger.debug("No data")
            # trackers will not have data associated. data_ind will be epmpty
            return trackList

        # Find valid data points : nan indicates missing data
        ValidDataLabel  = ~np.any(np.isnan(dataList), axis=0)
        valid_data_ind  = np.where(ValidDataLabel)[0]
        valid_data_num  = len(valid_data_ind)
       

        # Handle case of no valid data
        if valid_data_num < 1:
            logger.debug("No valid data for trackers")
            return trackList
        valid_data      = dataList[:,valid_data_ind]

        # Find trackers in tracking state and undefined tracks
        valid_track_states = np.zeros((TrackNum,1),dtype = bool)
        for k in range(TrackNum):
            valid_track_states[k] = trackList[k].check_valid() 

        valid_track_ind  = np.where(valid_track_states)[0]
        valid_track_num  = len(valid_track_ind)                   

        # Calculate distance metric for the valid trackers
        # Metric could be actual distance or probability / scaled by tracker covariance
        dist_metric = np.ones((valid_track_num, valid_data_num)) * 1e6
        for k in range(valid_track_num):
            dist_track          = trackList[valid_track_ind[k]].association_distance(valid_data)
            dist_metric[k, :]   = dist_track.reshape((1,-1))  # Assuming gaussian_prob is defined

        # Gating - the max distance. GateLevel should be compatible with measure distance
        DistLabels = dist_metric < GateLevel

        # Associate data with valid tracks
        for k in range(valid_track_num):
            ValidAssociatedInd                      = np.where(DistLabels[k, :])[0]
            trackList[valid_track_ind[k]].data_ind  = valid_data_ind[ValidAssociatedInd]

        # Find unassociated data and tracks
        nonassociated_data_ind = np.where(np.sum(DistLabels, axis=0) == 0)[0]
        nonassociated_data_len = len(nonassociated_data_ind)

        logger.debug(f"Unassociated points number: {nonassociated_data_len}") 

        UnAssocTrackInd = np.where(np.sum(DistLabels, axis=1) == 0)[0]
        UnAssocTrackLen = len(UnAssocTrackInd)

        logger.debug(f"Unassociated tracks number: {UnAssocTrackLen}")

        # Randomly assign UNDEFINED tracks with the rest of the data points
        nonvalid_track_ind  = np.where(~valid_track_states)[0]
        nonvalid_track_num  = len(nonvalid_track_ind) 

        # number of different data points and tracks should match
        new_assigned_num   = np.minimum(nonassociated_data_len,nonvalid_track_num)

        # Associate data with valid tracks
        for k in range(new_assigned_num):
            trackList[nonvalid_track_ind[k]].data_ind   = nonassociated_data_ind[k]

        logger.debug(f"New associated track number: {new_assigned_num}")
            
        return trackList        
    
    def track_update(self, trackList, dataList, par):
        """
        Performs track update using Kalman filter (PDAF optional).

        Args:
            trackList: List of track objects.
            dataList: 2D array containing measurement data (time x measurements).
            Par: Dictionary containing parameters.

        Returns:
            trackList: Updated list of track objects.
        """

        TrackNum    = par["TrackNum"]

        # next state 
        for i in range(TrackNum):

            # Extract track information
            track   = trackList[i]

            # compute next state
            state   = track.update_state()

            # extract associated data
            ydata   = track.extract_data(dataList)

            # Init state of the first time created
            ret     = track.init_state(ydata)            

            # Init velocity if data is OK and extract the relevant part
            ret     = track.init_velocity(ydata)

            # Kalman step
            ret     = track.update(ydata)
            
            # Update track state and life time
            ret     = track.update_statistics()

            # log likelihood management
            ret     = track.compute_likelihood()

            # back - no need
            trackList[i] = track
            
        
        self.time_counter    += 1
        # ... (rest of the code)
        return trackList


    def track_separation(self, trackList, dataList, par):
        """
        Identifies and discards tracks with similar histories.

        Args:
            trackList: List of track objects.
            dataList: 2D array containing measurement data (time x measurements).
            Par: Dictionary containing parameters.

        Returns:
            trackList: Updated list of track objects.
        """

        TrackNum = par["TrackNum"]
        HistGate = par["HistGateLevel"] * (par["Y1Bounds"][1] - par["Y1Bounds"][0]) * (par["Y2Bounds"][1] - par["Y2Bounds"][0])

        # Calculate pairwise distances between track histories
        HistDist = np.ones((TrackNum, TrackNum)) * 10000
        for i in range(TrackNum - 1):
            for j in range(i + 1, TrackNum):
                HistDist[i, j] = np.sum(np.std(trackList[i].history - trackList[j].history, axis=1))

        SameTracks = HistDist < HistGate

        # Identify valid tracks and their lifetimes
        ValidTrackLabel = np.zeros(TrackNum, dtype=bool)
        TracksLifeTime  = np.zeros(TrackNum)
        for i in range(TrackNum):
            if trackList[i].state != par["State_Undefined"]:
                ValidTrackLabel[i] = True
                TracksLifeTime[i]  = trackList[i].life_time

        ValidTrackInd = np.where(ValidTrackLabel)[0]
        ValidTrackLen = len(ValidTrackInd)
        if ValidTrackLen == 0:
            logger.info("Undefined States - check!!!")
            return trackList
        else:
            logger.info(f"Valid state tracks number: {ValidTrackLen}")

        # Sort valid tracks by lifetime
        sorted_ind      = np.argsort(TracksLifeTime[ValidTrackInd])[::-1]  # Descending order
        SortedTrackInd  = ValidTrackInd[sorted_ind]

        # Update SameTracks matrix with sorted indices
        SameTracks      = SameTracks[SortedTrackInd, :]
        SameTracks      = SameTracks[:,ValidTrackInd]

        # Loop through valid tracks
        for i in range(ValidTrackLen):
            SameTrackInd = np.where(SameTracks[i, :])[0]

            for j in range(len(SameTrackInd)):
                trackList[SameTrackInd[j]]["State"] = par["State_Undefined"]
                SameTracks[SameTrackInd[j], :] = 0

        return trackList

    def track_start(self, trackList, dataList, par):
        """
        Initializes new tracks based on unassociated data.

        Args:
            trackList   : List of track objects.
            dataList    : 2D x DataPoint Num array containing measurement data (time x measurements).
            par         : Dictionary containing parameters.

        Returns:
            trackList   : Updated list of track objects.
        """

        TrackNum            = par["TrackNum"]
        ValidStates         = par['State_List_Start']
        validData           = ~np.isnan(dataList)

        # Find unassociated data
        UnAssociatedData = np.all(~np.isnan(dataList), axis=0)

        # Find undefined tracks
        UnDefinedTrackLabel = np.zeros(TrackNum, dtype=bool)

        for i in range(TrackNum):
            for s in ValidStates:
                if trackList[i]["State"] == s :
                    UnAssociatedData[trackList[i]["DataInd"]] = False
                if trackList[i]["State"] == par["State_Undefined"]:
                    UnDefinedTrackLabel[i] = True        
        
        # for i in range(TrackNum):
        #     #ValidStates = [par["State_Track"]] + list(range(par["State_FirstInit"], par["State_LastInit"] + 1))
        #     if any(trackList[i]["State"] == s for s in ValidStates):
        #         UnAssociatedData[trackList[i]["DataInd"]] = False
        #     if trackList[i]["State"] == par["State_Undefined"]:
        #         UnDefinedTrackLabel[i] = True

        UnAssocDataInd = np.where(UnAssociatedData)[0]
        UnAssocDataLen = len(UnAssocDataInd)

        logger.debug(f"Unassociated points number: {UnAssocDataLen}")

        UnDefinedTrackInd = np.where(UnDefinedTrackLabel)[0]
        UnDefinedTrackLen = len(UnDefinedTrackInd)

        logger.debug(f"Undefined state tracks number: {UnDefinedTrackLen}")

        # Initialize new tracks
        if UnDefinedTrackLen == 0:
            logger.debug("No undefined tracks")
            return trackList

        if UnAssocDataLen < UnDefinedTrackLen:
            # More tracks than unassociated data
            dy1 = par["Y1Bounds"][1] - par["Y1Bounds"][0]
            dy2 = par["Y2Bounds"][1] - par["Y2Bounds"][0]
            RandLocY1 = np.random.rand(UnDefinedTrackLen - UnAssocDataLen) * dy1 + par["Y1Bounds"][0]
            RandLocY2 = np.random.rand(UnDefinedTrackLen - UnAssocDataLen) * dy2 + par["Y2Bounds"][0]
            RandLoc = np.vstack([RandLocY1, RandLocY2])
            DataLoc = np.hstack([RandLoc, dataList[:, UnAssocDataInd]])
        else:
            DataLoc = dataList[:, UnAssocDataInd[:UnDefinedTrackLen]]


        logger.info(f"{UnDefinedTrackLen} tracks are initiated and {UnAssocDataLen} from random location")

        # Initialize new tracks

        for i in range(UnDefinedTrackLen):
            xnew                                = np.zeros_like(trackList[UnDefinedTrackInd[i]]["x"])
            xnew[trackList[UnDefinedTrackInd[i]]["ObservInd"]] = DataLoc[:, i].reshape((-1,1))
            Pnew                                = trackList[UnDefinedTrackInd[i]]["Q"] * 10

            trackList[UnDefinedTrackInd[i]]["x"]        = xnew
            trackList[UnDefinedTrackInd[i]]["P"]        = Pnew
            trackList[UnDefinedTrackInd[i]]["State"]    = par["State_FirstInit"]
            trackList[UnDefinedTrackInd[i]]["LifeTime"] = 0
            trackList[UnDefinedTrackInd[i]]["LogLike"]  = par["GateLevel"]
            trackList[UnDefinedTrackInd[i]]["Hist"]     = np.zeros_like(trackList[UnDefinedTrackInd[i]]["Hist"])

        return trackList
   
    def track_print(self, trackList, par):
        "print tracking debug info nicely into columns"

        TrackNum             = par["TrackNum"]

        print_line           = '%4s |' %(str(self.time_counter))
        for k in range(TrackNum):
            print_line = print_line + ' %2s-%s |' %(str(trackList[k].state),str(trackList[k].life_time))  

        logger.info(print_line)

        return trackList

 
    def finish(self):
        # Close everything
        try:
            pass
        except:
            print('No window found')


    def tracking_demo(self):
        """
        Simulates a 2D tracking scenario using PDAF.
        """

        # Initialize parameters
        par       = self.init_parameters()

        # Initialize data,
        allData = self.init_data(par)

        # Initialize Kalman filter tracks
        trackList = self.init_tracks(par)

        # PDAF filtering loop
        for k in range(allData.shape[2]):
            # Get the data for time k
            dataList = allData[:, :, k]

            # Show the current state (optional)
            #self.show_tracks_and_data(trackList, dataList, par)  # Assuming this function is defined

            # Data-track association
            trackList = self.track_association(trackList, dataList, par)  # Assuming this function is defined

            # Track update
            trackList = self.track_update(trackList, dataList, par)  # Assuming this function is defined

            # Track separation
            trackList = self.track_separation(trackList, dataList, par)  # Assuming this function is defined

            # Start new tracks
            trackList = self.track_start(trackList, dataList, par)  # Assuming this function is defined

            # print tracks info
            trackList = self.track_print(trackList, par)

            # Record data (optional)
            # Record = Structure_PDAF_Record(Record, TrackList, DataList)  # Assuming this function is defined

        # Show final results
        #self.show_tracks_and_data(trackList, dataList, par)  # Assuming this function is defined            

# --------------------------------
#%% Tests
class TestPDAF(unittest.TestCase):

    def test_create(self):
        "check create and data generation"
        p       = PDAF()
        d       = DataGenerator()
        s       = DataDisplay()
        
        par     = p.init_parameters()
        ydata,t = d.init_data(par)   
        ax      = s.init_show()

        p.finish()
        self.assertTrue(len(ydata) > 0) 

    def test_show_data(self):
        "check create and data generation"
        p       = PDAF()
        d       = DataGenerator()
        s       = DataDisplay()
        
        par     = p.init_parameters()
        ydata,t = d.init_data(par)    
        tlist   = p.init_tracks(par)
        ax      = s.init_show(par)

        # PDAF filtering loop
        for k in range(ydata.shape[1]):
            # Get the data for time 2 x k x point_num
            dlist    = ydata[:, k, :].squeeze()
            s.show_info(tlist, dlist)
            time.sleep(0.1)

        p.finish()
        self.assertTrue(len(ydata) > 0) 

    def test_association(self):
        "check tracker and data association"
        p       = PDAF()
        d       = DataGenerator()
        s       = DataDisplay()
        
        par     = d.init_scenario(1)
        ydata,t = d.init_data(par)    
        tlist   = p.init_tracks(par)
        ax      = s.init_show(par)

        # PDAF filtering loop
        for k in range(ydata.shape[1]):
            # Get the data for time 2 x k x point_num
            dlist       = ydata[:, k, :].reshape((2,-1))
            tlist       = p.track_association(tlist, dlist, par)
            tlist       = p.track_print(tlist, par)
            s.show_info(tlist, dlist)
            time.sleep(0.1)

        p.finish()
        self.assertTrue(len(ydata) > 0)    
        
    def test_update(self):
        "check tracker update and data association"
        p       = PDAF()
        d       = DataGenerator()
        s       = DataDisplay()
        
        par     = d.init_scenario(1)
        ydata,t = d.init_data(par)    
        tlist   = p.init_tracks(par)
        ax      = s.init_show(par)

        # PDAF filtering loop
        for k in range(ydata.shape[1]):
            # Get the data for time 2 x k x point_num
            dlist       = ydata[:, k, :].reshape((2,-1))
            tlist       = p.track_association(tlist, dlist, par)
            tlist       = p.track_update(tlist, dlist, par)
            tlist       = p.track_print(tlist, par)
            
            s.show_info(tlist, dlist)
            time.sleep(0.1)

        p.finish()
        self.assertTrue(len(ydata) > 0)          

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestPDAF("test_create")) # ok
    #suite.addTest(TestPDAF("test_show_data")) # ok
    #suite.addTest(TestPDAF("test_association")) 
    suite.addTest(TestPDAF("test_update")) 

    
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()
    #RunApp('iir',43).run()    

