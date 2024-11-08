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
import matplotlib.pyplot as plt
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

        self.tprint(f'Created')

    def init_parameters(self):
        """
        Initializes different parameters of the algorithm.

        Returns:
            Par: A dictionary containing the algorithm parameters.
        """
        par                 = config_parameters()

        track_num           = par['TrackNum']
        self.tprint(f'Track number : {track_num}')
        return par     

    def create_point_cover_2d(self, par):
        "creates evently distributed cover of points in the spce"
        # Distribute trackers evenly over the measurement space
        dy1        = par["Y1Bounds"][1] - par["Y1Bounds"][0]
        dy2        = par["Y2Bounds"][1] - par["Y2Bounds"][0]
        TrackNumY1 = max(1, int(np.sqrt(dy1 / dy2 * par["TrackNum"])))
        TrackNumY2 = int(np.ceil(par["TrackNum"] / TrackNumY1))

        yy1, yy2 = np.meshgrid(
            np.linspace(par["Y1Bounds"][0] + dy1 / TrackNumY1 / 2, par["Y1Bounds"][1], TrackNumY1),
            np.linspace(par["Y2Bounds"][0] + dy2 / TrackNumY2 / 2, par["Y2Bounds"][1], TrackNumY2)
        )
        cover_data = np.hstack((yy1, yy2)).T
        return cover_data                

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

        centerData = self.create_point_cover_2d(par)

        # Initialize the track list
        trackList = []
        for i in range(track_num):

            track       = TrackingObject(par, i+1)
            track.init_state(centerData[:,i])

            trackList.append(track)

        return trackList
    
    def track_association(self, trackList, dataList, par):
        """
        Associates data points with tracks.

        Args:
            trackList: Kalman structure list and more
            dataList:  2 x DataPointsNum contains the relevant data for time t
            Par: Dictionary containing parameters.

        Returns:
            trackList: Updated list of track objects.
        """

        GateLevel   = par["GateLevel"]
        TrackNum    = par["TrackNum"]

        # Check for undefined tracks
        for i in range(TrackNum):
            if not trackList[i].check_valid():
                raise ValueError(f"Undefined track {trackList[i].id}")

        # Find valid data points
        ValidDataLabel  = ~np.any(np.isnan(dataList), axis=0)
        ValidDataInd    = np.where(ValidDataLabel)[0]
        ValidDataNum    = len(ValidDataInd)

        # Handle case of no valid data
        if ValidDataNum == 0:
            self.tprint("No valid data")
            ResolvedValidDataNum = 1 # to prevent zero columns DistM matrix
        else:
            ResolvedValidDataNum = ValidDataNum


        # Calculate distance matrix
        DistM = np.ones((TrackNum, ResolvedValidDataNum)) * 1e6
        for i in range(TrackNum):
            dist_track  = trackList[i].association_distance(dataList)
            DistM[i, :] = dist_track.reshape((1,-1))  # Assuming gaussian_prob is defined

        # Gating
        DistLabels = DistM < GateLevel

        # Associate data with tracks
        for i in range(TrackNum):
            ValidAssociatedInd      = np.where(DistLabels[i, :])[0]
            trackList[i].data_ind   = ValidDataInd[ValidAssociatedInd]

        # Find unassociated data and tracks
        UnAssocDataInd = np.where(np.sum(DistLabels, axis=0) == 0)[0]
        UnAssocDataLen = len(UnAssocDataInd)

        self.tprint(f"Unassociated points number: {UnAssocDataLen}")

        UnAssocTrackInd = np.where(np.sum(DistLabels, axis=1) == 0)[0]
        UnAssocTrackLen = len(UnAssocTrackInd)

        self.tprint(f"Unassociated tracks number: {UnAssocTrackLen}")
            
        return trackList        
    
    def track_update(trackList, dataList, par):
        """
        Performs track update using a Kalman filter (PDAF optional).

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
            self.tprint("Undefined States - check!!!")
            return trackList
        else:
            self.tprint(f"Valid state tracks number: {ValidTrackLen}")

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
        if par["ShowOn"]:
            self.tprint(f"Unassociated points number: {UnAssocDataLen}")

        UnDefinedTrackInd = np.where(UnDefinedTrackLabel)[0]
        UnDefinedTrackLen = len(UnDefinedTrackInd)
        if par["ShowOn"]:
            self.tprint(f"Undefined state tracks number: {UnDefinedTrackLen}")

        # Initialize new tracks
        if UnDefinedTrackLen == 0:
            if par["ShowOn"]:
                self.tprint("No undefined tracks")
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

        if par["ShowOn"]:
            self.tprint(f"{UnDefinedTrackLen} tracks are initiated and {UnAssocDataLen} from random location")

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
    
#    def  show_init(self, ax = None, fig = None):
#        # init 3D scene
#        if ax is None or fig is None:
#            fig = plt.figure(1)
#            plt.clf() 
#            plt.ion() 
#            #fig.canvas.set_window_title('3D Scene')
#            ax = fig.add_subplot(projection='3d')
#            fig.tight_layout()
#            
#            #ax.set_proj_type('ortho')
#            
#            #self.ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#            #self.ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#            #self.ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#
#        #ax.set_aspect("equal")
#        plt.title('Data Points & Tracks')
#        plt.xlabel('X1')
#        plt.ylabel('X2')
#        ax = plt.gca()
#        ax.set_xlim([self.params["Y1Bounds"][0], self.params["Y1Bounds"][1]])
#        ax.set_ylim([self.params["Y2Bounds"][0], self.params["Y2Bounds"][1]])
#
#        #ax.set_xlabel('x')
#        #ax.set_ylabel('y')
#        #ax.set_zlabel('z')
#        
#        #ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#        #ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#        #ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
#        
#        
#        #ax.set_title('Object Visualization')
#        plt.show()
#        self.Print('Scene rendering is done')
#        
#        self.fig    = fig
#        self.ax     = ax
#        return ax 

#    def show_tracks_and_data(self, trackList, dataList, par):
#        """
#        Visualizes the data points and tracks.
#
#        Args:
#            trackList: List of track objects.
#            dataList: 2D array containing measurement data (time x measurements).
#            Par: Dictionary containing parameters.
#        """
#
#        ShowFigNum  = 1
#        #AxisSc      = [par["Y1Bounds"][0], par["Y2Bounds"][0], par["Y1Bounds"][1], par["Y2Bounds"][1]]
#        SmallShift  = 5e-3
#        NumSigma    = np.sqrt(par["GateLevel"])
#
#        # Plot data points
#        plt.figure(ShowFigNum)
#        plt.plot(dataList[0, :], dataList[1, :], 'b.')
#        #plt.axis(AxisSc)
#        plt.title('Data Points & Tracks')
#        plt.xlabel('X1')
#        plt.ylabel('X2')
#        ax = plt.gca()
#        ax.set_xlim([par["Y1Bounds"][0], par["Y1Bounds"][1]])
#        ax.set_ylim([par["Y2Bounds"][0], par["Y2Bounds"][1]])
#        
#        # Plot tracks
#        TrackNum            = len(trackList)
#
#        for i in range(TrackNum):
#
#            y, S    = trackList[i].predict()
#            u, s, v = np.linalg.svd(S)
#            elipse  = u @ np.diag(np.sqrt(s)) @ np.vstack((np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))))
#
#            # do not show certain states
#            #if not any(trackList[i]["State"] == s for s in ValidStatesForShow):
#            #    y = np.array([[np.nan], [np.nan]])
#
#            plt.plot(elipse[0, :] + y[0], elipse[1, :] + y[1], 'r')
#            plt.text(y[0] + SmallShift, y[1], str(i), fontsize=8)
#
#        plt.draw()
#        plt.pause(0.1)  # Update the plot
#        #plt.clf()       
 
    def finish(self):
        # Close everything
        try:
            #cv.destroyWindow(self.estimator_name) 
            pass
        except:
            print('No window found')

    def tprint(self, txt = '', level = 'I'):
        txt = 'PDF : '+ txt
        if level == "I":
            logger.info(txt)
        elif level == "W":
            logger.warning(txt)
        elif level == "E":
            logger.error(txt)
        else:
            logger.info(txt)

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
            self.show_tracks_and_data(trackList, dataList, par)  # Assuming this function is defined

            # Data-track association
            trackList = self.track_association(trackList, dataList, par)  # Assuming this function is defined

            # Track update
            trackList = self.track_update(trackList, dataList, par)  # Assuming this function is defined

            # Track separation
            trackList = self.track_separation(trackList, dataList, par)  # Assuming this function is defined

            # Start new tracks
            trackList = self.track_start(trackList, dataList, par)  # Assuming this function is defined

            # Record data (optional)
            # Record = Structure_PDAF_Record(Record, TrackList, DataList)  # Assuming this function is defined

        # Show final results
        self.show_tracks_and_data(trackList, dataList, par)  # Assuming this function is defined            

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
        
        par     = d.init_scenario(9)
        ydata,t = d.init_data(par)    
        tlist   = p.init_tracks(par)
        ax      = s.init_show(par)

        # PDAF filtering loop
        for k in range(ydata.shape[1]):
            # Get the data for time 2 x k x point_num
            dlist       = ydata[:, k, :].reshape((2,-1))
            tlist       = p.track_association(tlist, dlist, par)
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
    suite.addTest(TestPDAF("test_association")) 

    
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()
    #RunApp('iir',43).run()    

