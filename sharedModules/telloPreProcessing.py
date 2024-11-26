'''
This file is intended to hold code for counting the amount of detections within a given region.
It will be held within a folder outside of the main file so it needs to be an importable class.
It needs to return an integer that counts the amount of people within the the 720x720 area.
Furthermore this class can do more preprocessing for us such as cosine distance metric, cropping the photo, etc...
'''

import tellopy
import cv2


class PreProccessing:

    def __init__():
        '''
        This is the constructor for the Class.
        Need to identify what parameters are needed for the following methods:
            tellopy is our mock control algorithm for the drone
                -https://djitellopy.readthedocs.io/en/latest/tello/
            cv2 for image processing
        '''



    
    def countRegion():
        '''
        This method will count how many detections were made within the 720x720 area.
        This will allow us to dynamically create a list of detections every frame.
        '''


    def cosineDistance():
        '''
        We will have a LiDAR for height data so we can assume h = 15 ft and theta = 30.
        '''
        cosdis = 0

        return cosdis

