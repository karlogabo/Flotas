#!/usr/bin/env python
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import keyboard

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y16, 30)

# Start streaming
pipeline.start(config)

#Asks for the user ID
id = str(raw_input("Introduzca la cedula: "))

#Creates folder for the user
path = ("/home/innovacion/Bags/" + id )
os.mkdir(path)
count = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infrared_frame = frames.get_infrared_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        infrared_image = np.asanyarray(infrared_frame.get_data())

        # Stack both images horizontally
        images = np.hstack((infrared_image, depth_image))

        #Take the pictures
        k = cv2.waitKey(1)
        if k%256 == 32:
            s = str(count)
            print("Empece la foto", s)
            path_infra = (path + "/" + s + "infra.png")
            path_rgb = (path + "/" + s + "rgb.png")
            path_depth = (path + "/" + s + "depth.png")
            cv2.imwrite(path_infra, infrared_image)
            cv2.imwrite(path_rgb, color_image)
            cv2.imwrite(path_depth, depth_image)
            count += 1
            print("Termine la toma", s)

        # Show depth and infrared images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        #Show color image
        cv2.namedWindow('RealSenseb', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSenseb', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()
