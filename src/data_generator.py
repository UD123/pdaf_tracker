#!/usr/bin/env python

'''
Utility functions for PDAF
==================



Usage:

Environemt : 

Install : 


'''

import numpy as np
import unittest

import matplotlib.pyplot as plt

#import logging 
#logger = logging.getLogger('pdaf')
from utils import logger



#%% Main
class DataGenerator:
    "class to create images and points"
    def __init__(self):

        self.frame_size = (640,480)
        self.img        = None

        self.tprint('Created')

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
            SlopeV = 5
            tmp = 1 / (1 + np.exp(-SlopeV * (t - Time / 2)))
            y = np.array([tmp, t / Time])

        elif TrajType == 3:  # Spiral
            R = 0.5 + 0.5 * np.sin(2 * np.pi / Time * t)
            Ang = 2 * np.pi * t
            y = 0.5 * np.array([R * np.cos(Ang), R * np.sin(Ang)]) + 0.5

        elif TrajType == 4:  # 8
            fx = 0.5
            fy = 1
            y = 0.5 * np.array([np.cos(2 * np.pi * fx * t), np.sin(2 * np.pi * fy * t)]) + 0.5

        elif TrajType == 5:  # Exponential
            SlopeV = 5
            y = np.array([1 - np.exp(-t / Time * SlopeV), t / Time])

        elif TrajType == 6:  # Triangle
            y = np.array([t / Time, np.triang(N).T])

        elif TrajType == 7:  # Rising
            SlopeV = 1
            N3 = round(N / 3)
            y = np.array([t / Time, np.concatenate([np.zeros(N3), np.linspace(0, SlopeV, N3), np.ones(N - 2 * N3) * SlopeV])])

        elif TrajType == 8:  # Non-moving random point
            tmp = np.random.rand(1, 2)
            y = np.repeat(tmp, N, axis=0)

        elif TrajType == 9:  # Straight line from left upper to right lower
            LeftH = 0.95
            RightH = 0.05
            x = np.linspace(LeftH, RightH, N).reshape((-1,1))
            y = np.hstack((t / Time, x))

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
 
    def add_noise(self, img_gray, noise_percentage = 0.01):
        "salt and pepper noise"
        if noise_percentage < 0.001:
            return img_gray


        # # Get the image size (number of pixels in the image).
        # img_size = img_gray.size

        # # Set the percentage of pixels that should contain noise
        # #noise_percentage = 0.1  # Setting to 10%

        # # Determine the size of the noise based on the noise precentage
        # noise_size = int(noise_percentage*img_size)

        # # Randomly select indices for adding noise.
        # random_indices = np.random.choice(img_size, noise_size)

        # # Create a copy of the original image that serves as a template for the noised image.
        img_noised = img_gray.copy()

        # # Create a noise list with random placements of min and max values of the image pixels.
        # #noise = np.random.choice([img_gray.min(), img_gray.max()], noise_size)
        # noise = np.random.choice([-10, 10], noise_size)

        # # Replace the values of the templated noised image at random indices with the noise, to obtain the final noised image.
        # img_noised.flat[random_indices] += noise
        
        self.tprint('adding noise')
        return img_noised
 
    def create_point_cover_2d(self, Par):
        "creates evently districbuted cover of points"
        # Distribute trackers evenly over the measurement space
        dy1 = Par["Y1Bounds"][1] - Par["Y1Bounds"][0]
        dy2 = Par["Y2Bounds"][1] - Par["Y2Bounds"][0]
        TrackNumY1 = max(1, int(np.sqrt(dy1 / dy2 * Par["TrackNum"])))
        TrackNumY2 = int(np.ceil(Par["TrackNum"] / TrackNumY1))

        yy1, yy2 = np.meshgrid(
            np.linspace(Par["Y1Bounds"][0] + dy1 / TrackNumY1 / 2, Par["Y1Bounds"][1], TrackNumY1),
            np.linspace(Par["Y2Bounds"][0] + dy2 / TrackNumY2 / 2, Par["Y2Bounds"][1], TrackNumY2)
        )
        cover_data = np.vstack([yy1, yy2]).T
        return cover_data

    def show_points_2d(self, y, t):
        "display in 3D"
        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d')

        ts,xs,ys       = t.reshape((-1,1)), y[0,:].reshape((-1,1)), y[1,:].reshape((-1,1))
        ax.plot(xs, ys)
        
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
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
        p           = DataGenerator()
        y,t,dt      = p.generate_trajectories(1)

        p.show_points_2d(y, t)
        self.assertFalse(p.img is None)


if __name__ == '__main__':
    #print(__doc__)

    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestDataGenerator("test_generate_trajectories"))
 
   
    runner = unittest.TextTestRunner()
    runner.run(suite)

