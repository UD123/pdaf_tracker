#!/usr/bin/env python

'''
Utility functions for PDAF
==================

Usage:

Environment : 

Install : 


'''

import numpy as np
import unittest
import matplotlib.pyplot as plt

from utils import logger, config_parameters



#%% Main
class DataGenerator:
    "class to create images and points"
    def __init__(self):

        self.params      = config_parameters()

        logger.debug('Created')

    def init_scenario(self, scene_type = 1):
        "changes parameters to create different scenes"
        "TrajIndex - list that contains different trajectories "
        "PointNum  - must be bigger than TrajIndex list. if PointNum > len(TrajIndex) may create some clutter points"
        Par         = self.params
        if scene_type == 1: # fixed point
            Par["TrajIndex"]    = [8] 
            Par['PointNum']     = 1
        elif scene_type == 2:
            Par["TrajIndex"]    = [7] 
            Par['PointNum']     = 1
        elif scene_type == 3:
            Par["TrajIndex"]    = [7] 
            Par['PointNum']     = 2       
        elif scene_type == 4:
            Par["TrajIndex"]    = [2,7] 
            Par['PointNum']     = 3     
        elif scene_type == 5:
            Par["TrajIndex"]    = [3,4] 
            Par['PointNum']     = 2                 
        elif scene_type == 6: # clutter
            Par["TrajIndex"]    = [5] 
            Par['PointNum']     = 3                     
        elif scene_type == 7: # noise
            Par["TrajIndex"]    = [1] 
            Par['PointNum']     = 1  
            Par['Nv']           = 0.05
        elif scene_type == 8: # missing data
            Par["TrajIndex"]    = [1] 
            Par['PointNum']     = 1  
            Par['NaNDensity']   = 0.1  
        elif scene_type == 9: # straight line
            Par["TrajIndex"]    = [9] 
            Par['PointNum']     = 2  
            Par['Time']         = 3  

             
        else: 
            # default
            pass
            
        Par["TrajNum"]      = t_num = len(Par["TrajIndex"])  
        p_num               = Par['PointNum']           
        self.params         = Par
        logger.debug(f"Generating scene {scene_type} : trajectories {t_num}, points {p_num} ")
        return Par

    def generate_trajectories(self, TrajType=1, dT=1/30, Time=3):
        """
        Generates 2D trajectories for tracking.

        Args:
            TrajType: Trajectory type (integer).
            dT: Time step.
            Time: Total simulation time.

        Returns:
            y: 2D array of trajectory points.
            t: Time vector.
            dT: Time step.
        """

        t = np.arange(0, Time + dT, dT).reshape((-1,1))
        N = len(t)

        if TrajType == 1:  # Circle
            fx = 0.4
            y  = 0.5 * np.hstack((np.cos(2 * np.pi * fx * t), np.sin(2 * np.pi * fx * t))) + 0.5

        elif TrajType == 2:  # Jump in x
            SlopeV  = 5
            tmp     = 1 / (1 + np.exp(-SlopeV * (t - Time / 2)))
            y       = np.hstack((tmp, t / Time))

        elif TrajType == 3:  # Spiral
            R       = 0.5 + 0.5 * np.sin(2 * np.pi / Time * t)
            Ang     = 2 * np.pi * t
            y       = 0.5 * np.hstack((R * np.cos(Ang), R * np.sin(Ang))) + 0.5

        elif TrajType == 4:  # 8
            fx      = 0.5
            fy      = 1
            y = 0.5 * np.hstack((np.cos(2 * np.pi * fx * t), np.sin(2 * np.pi * fy * t))) + 0.5

        elif TrajType == 5:  # Exponential
            SlopeV = 5
            y       = np.hstack((1 - np.exp(-t / Time * SlopeV), t / Time))

        elif TrajType == 6:  # Triangle
            y       = np.array([t / Time, np.triang(N).T])

        elif TrajType == 7:  # Rising
            SlopeV  = 1
            N3      = round(N / 3)
            x       = np.hstack((np.zeros((1,N3)), np.linspace(0, SlopeV, N3).reshape((1,N3)), np.ones((1, N - 2 * N3)) * SlopeV)).T
            y       = np.hstack((t / Time, x[:N]))

        elif TrajType == 8:  # Non-moving random point
            tmp     = np.random.rand(1, 2)
            y       = np.repeat(tmp, N, axis=0)

        elif TrajType == 9:  # Straight line from left upper to right lower
            LeftH   = 0.95
            RightH  = 0.05
            x       = np.linspace(LeftH, RightH, N).reshape((-1,1))
            y       = np.hstack((t / Time, x))

        elif TrajType == 10:  # Straight line from left lower to right upper
            LeftH = 0.05
            RightH = 0.95
            x = np.linspace(LeftH, RightH, N).reshape((-1,1))
            y = np.hstack((t / Time, x))

        elif TrajType == 11:  # Straight line from left upper to right lower
            LeftH = 0.75
            RightH = 0.25
            x = np.linspace(LeftH, RightH, N).reshape((-1,1))
            y = np.hstack((t / Time, x))

        elif TrajType == 12:  # Straight line from left lower to right upper
            LeftH = 0.25
            RightH = 0.75
            x = np.linspace(LeftH, RightH, N).reshape((-1,1))
            y = np.hstack((t / Time, x))

        else:
            raise ValueError("Unknown trajectory type")
        
        # row vectors DxT
        y = y.T
        t = t.T

        return y, t, dT 
 
    def init_data(self, par = None):
        """
        Initializes data for multiple tracking models.

        Args:
            Par: A dictionary containing parameters:
                - TrajIndex: Trajectory indices
                - Nv: Noise variance
                - PointNum: Number of points
                - NaNDensity: Density of missing points
                - dT: Time step
                - Time: Total time

        Returns:
            yc  : 2 x Time x PointNum -  numpy array containing the initialized data.
        """
        par         = self.params if par is None else par

        # Check for missing parameters
        if not all(field in par for field in ['TrajIndex', 'Nv', 'PointNum', 'NaNDensity']):
            raise ValueError("Missing required fields in Par.")

        # Generate trajectories and clutter
        y, t, dT    = self.generate_trajectories(par['TrajIndex'][0], par['dT'], par['Time'])  # Assuming Generate2DTrajectories is defined
        yc          = np.random.rand(y.shape[0],  y.shape[1], par['PointNum'],)

        # Ensure enough points for trajectories
        if par['PointNum'] < par['TrajNum']:
            logger.debug("There are more trajectories than points.")
            par['PointNum'] = par['TrajNum'] + 1

        # Random permutation
        RandPermTraj = np.random.permutation(par['PointNum'])

        # Initialize trajectories
        for i in range(par['TrajNum']):
            ytmp, t, dT                  = self.generate_trajectories(par['TrajIndex'][i], par['dT'], par['Time'])  # Assuming Generate2DTrajectories is defined
            #yc[:,:, RandPermTraj[i]]    = ytmp #.reshape((-1,2))
            yc[:,:, i]    = ytmp


        # Add noise
        yc             += np.random.randn(*yc.shape) * par['Nv']

        # Generate missing points
        miss_ind        = np.random.rand(*yc.shape) < par['NaNDensity']
        yc[miss_ind]    = np.nan

        return yc, t        

    def show_points_2d(self, y, t):
        "display in 3D"
        traj_num       = y.shape[2]
        

        fig             = plt.figure()
        ax              = fig.add_subplot(projection='3d')

        ts              = t.reshape((-1,1))
        for k in range(traj_num):

            xs,ys       = y[0,:,k].reshape((-1,1)), y[1,:,k].reshape((-1,1))
            ax.scatter(xs, ys, marker='.',color='C'+str(k))


        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        #ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', 'box')
        plt.show()        

    def tprint(self, txt = '', level = 'I'):
        txt = 'DAT : ' + txt
        if level == "I":
            logger.info(txt)
        elif level == "W":
            logger.warning(txt)
        elif level == "E":
            logger.error(txt)
        else:
            logger.info(txt)
   


# ----------------------
#%% Tests
class TestDataGenerator(unittest.TestCase):

    def test_generate_trajectories(self):
        "shows a simple "
        d           = DataGenerator()
        y,t,dt      = d.generate_trajectories(1)

        d.show_points_2d(y, t)
        self.assertFalse(d.params is None)

    def test_init_data(self):
        "init multipe trajectories "
        d           = DataGenerator()
        par         = d.init_scenario(9) # 1,2,3,4,5,6,7,8
        y,t         = d.init_data(par)

        d.show_points_2d(y, t)
        self.assertFalse(d.params is None)

if __name__ == '__main__':
    #print(__doc__)

    #unittest.main()
    suite = unittest.TestSuite()
    #suite.addTest(TestDataGenerator("test_generate_trajectories")) # ok
    suite.addTest(TestDataGenerator("test_init_data"))
 
   
    runner = unittest.TextTestRunner()
    runner.run(suite)

