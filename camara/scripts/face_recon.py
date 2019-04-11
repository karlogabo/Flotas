#!/usr/bin/env python
import rospy
import numpy as np
import dlib
import cv2
from face_recognition.msg import Faces
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import os
import sys

bridge = CvBridge()
enable_img = False
enable_infra = False

class Face_recon(object):
    def __init__(self):

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'shape_predictor_68_face_landmarks.dat'))

        """Subscribers"""
        rospy.Subscriber("/camera/depth/image_rect_raw" , Image , self.callback_depth,queue_size=1)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image,  self.callback_infra,queue_size=1)

        """Publishers"""
        self.pub_detection= rospy.Publisher("head", Image)


    def visualizacion(self,frame):

        cv2.imshow("Frame",frame)
        cv2.waitKey(1)

    def callback_depth(self,datos):

        #Callback for acquiring depth data from the realsense
        global enable_img
        enable_img = True
        self.current_depth = datos

    def callback_infra(self,datos):

        #Callback for acquiring depth data from the realsense
        global enable_infra
        enable_infra = True
        self.current_infra = datos

    def main(self):

        try:
            self.depth = bridge.imgmsg_to_cv2(self.current_depth,desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        try:
            self.rojo = bridge.imgmsg_to_cv2(self.current_infra,desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        self.rojo = cv2.equalizeHist(self.rojo)
        rects = self.detector(self.rojo, 0)

        if not rects:

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (15,20)
            fontScale = 0.65
            fontColor = (255,0,0)
            lineType  = 1
            cv2.putText(self.rojo,'NO DETECTION!', bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            self.visualizacion(self.rojo)

        else:
            rect = rects.pop()
            self.roi_head = self.rojo[rect.top():rect.bottom(),rect.left():rect.right()]
            self.pub_detection.publish(bridge.cv2_to_imgmsg(self.roi_head, encoding="passthrough"))

if __name__ == '__main__':

    rospy.init_node('listener',anonymous=True)

    cam = Face_recon()
    rospy.loginfo('Node started')
    from time import time

    t = time()
    while not rospy.is_shutdown():
        print(time()-t)
        t = time()
        if enable_img is True and enable_infra is True:
            cam.main()
    rospy.spin()
