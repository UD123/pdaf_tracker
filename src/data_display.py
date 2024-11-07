
'''

PDAF Data and Tracking Display Manager

==================

Shows data points in real time along with trakers and histories.


Usage:

Environment : 

Install : 
    See README.md


'''

import numpy as np
import time

import numpy as np
import unittest
import matplotlib.pyplot as plt
# disable vefore compile
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D

from utils import logger, config_parameters


# --------------------------------
#%% Main
class DataDisplay:

    def __init__(self, par = None):

        self.params      = config_parameters() if par is None else par

        # show
        self.fig        = None
        self.plt        = None
        self.ax         = None    

        self.h_data     = None        
        self.h_pose     = None        
        self.h_text     = None
        self.h_circle   = None        
        self.h_history  = None            

        self.tprint(f'Created')
      
    def show_init(self, par = None):
        # init 2D/3D scene
        par         = self.params if par is None else par
        fig_num     = 1
        track_num   = par["TrackNum"]
        small_shift = 5e-3

        # init figure
        fig         = plt.figure(fig_num)
        #plt.clf() 
        plt.ion()    
        #fig.tight_layout()  
        #ax = fig.add_subplot(projection='3d')  

        # Plot data points for handler
        h_data,     = plt.plot([0, 1], [0, 1], 'b.')
        ax          = plt.gca()

        # plot tracker positions
        h_pose      = []
        for k in range(track_num):
            h,  = ax.plot([0], [0],marker='o',color='C'+str(k))
            h_pose.append(h)

        # plot tracker names
        h_text      = []
        for k in range(track_num):
            h  = ax.text(0+small_shift ,1, str(k), fontsize=8)
            h_text.append(h)

        # plot tracker uncertainty circles
        h_circle      = []
        for k in range(track_num):
            h,  = ax.plot([0, 0], [0,1], color='r')
            h_circle.append(h)   

        # plot tracker uncertainty circles
        h_history      = []
        for k in range(track_num):
            h,  = ax.plot([0, 0], [0, 0], color='g')
            h_history.append(h)                     
        
        plt.title('Data Points & Tracks')
        plt.xlabel('X1')
        plt.ylabel('X2')

        
        ax.set_xlim([par["Y1Bounds"][0], par["Y1Bounds"][1]])
        ax.set_ylim([par["Y2Bounds"][0], par["Y2Bounds"][1]])

        plt.draw()
        #plt.pause(0.1)  # Update the plot
        plt.show()
        
        self.h_data = h_data        
        self.h_pose = h_pose        
        self.h_text = h_text
        self.h_circle = h_circle        
        self.h_history = h_history


        self.fig    = fig
        self.plt    = plt
        self.ax     = ax
        self.tprint('Scene rendering is done')
        return ax 
    
    def draw_data(self, dataList):
        "assuming that everythong is initialized - show data"
        if dataList is None:
            return

        self.h_data.set_data(dataList[0, :], dataList[1, :])

    def draw_track_pose(self, trackList):
        "assuming that everythong is initialized"
        if trackList is None:
            return

        track_num   = self.params["TrackNum"]   

        # plot tracker positions
        for k in range(track_num):
            self.h_pose[k].set_data(trackList[0, :], trackList[1, :])        

    def show_info(self, trackList = None, dataList = None):
        "displays tracks and data"

        self.draw_data(dataList)
        self.draw_track_pose(trackList)

        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events() 

        self.plt.draw()
        self.plt.pause(0.1)

    def show_tracks_and_data(self, trackList, dataList, par = None):
        """
        Visualizes the data points and tracks.

        Args:
            trackList: List of track objects.
            dataList: 2D array containing measurement data (time x measurements).
            Par: Dictionary containing parameters.
            
        """
        par         = self.params if par is None else par

        ShowFigNum  = 1
        #AxisSc      = [par["Y1Bounds"][0], par["Y2Bounds"][0], par["Y1Bounds"][1], par["Y2Bounds"][1]]
        SmallShift  = 5e-3
        NumSigma    = np.sqrt(par["GateLevel"])

        # Plot data points
        plt.figure(ShowFigNum)
        plt.plot(dataList[0, :], dataList[1, :], 'b.')
        #plt.axis(AxisSc)
        plt.title('Data Points & Tracks')
        plt.xlabel('X1')
        plt.ylabel('X2')
        ax = plt.gca()
        ax.set_xlim([par["Y1Bounds"][0], par["Y1Bounds"][1]])
        ax.set_ylim([par["Y2Bounds"][0], par["Y2Bounds"][1]])
        
        # Plot tracks
        TrackNum            = len(trackList)

        for i in range(TrackNum):

            y, S    = trackList[i].predict()
            u, s, v = np.linalg.svd(S)
            elipse  = u @ np.diag(np.sqrt(s)) @ np.vstack((np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100))))

            # do not show certain states
            #if not any(trackList[i]["State"] == s for s in ValidStatesForShow):
            #    y = np.array([[np.nan], [np.nan]])

            plt.plot(elipse[0, :] + y[0], elipse[1, :] + y[1], 'r')
            plt.text(y[0] + SmallShift, y[1], str(i), fontsize=8)

        plt.draw()
        plt.pause(0.1)  # Update the plot
        #plt.clf()       
 
    def finish(self):
        # Close everything
        #plt.show()
        try:
            #cv.destroyWindow(self.estimator_name) 
            pass
        except:
            print('No window found')

    def tprint(self, txt = '', level = 'I'):
        txt = 'DSP : '+ txt
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
class TestDataDisplay(unittest.TestCase):

    def test_create(self):
        "check create and data show init"
        d       = DataDisplay()
        ax      = d.show_init()
        d.finish()
        self.assertFalse(ax is None) 

    def test_show_data(self):
        "check create and data show"
        d       = DataDisplay()
        ax      = d.show_init()
        trackP  = None
        for k in range(10):
            dataP   = np.random.rand(2,10)
            d.show_info(trackP, dataP)
        d.finish()
        self.assertFalse(ax is None) 

      

# --------------------------------
#%% Run Test
def RunTest():
    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestDataDisplay("test_create")) # ok
    suite.addTest(TestDataDisplay("test_show_data")) 

    
    
    runner = unittest.TextTestRunner()
    runner.run(suite)

if __name__ == '__main__':
    #print(__doc__)

    RunTest()
    #RunApp('iir',43).run()    
