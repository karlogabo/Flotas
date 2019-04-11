#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
import imutils
import dlib
import sys
import os
from cv_bridge import CvBridge, CvBridgeError

enable_infra = False
bridge = CvBridge()


class Modelo(object):

    def __init__(self):

        """ Subscribers """
        rospy.Subscriber("/camera/color/image_raw", Image,  self.callback_infra,queue_size=1)

        """ Node Parameters """
        protocol =  (os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'deploy.prototxt.txt'))
        model = (os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'res10_300x300_ssd_iter_140000.caffemodel'))
        self.net =  cv2.dnn.readNetFromCaffe(protocol, model)

    def callback_infra(self,datos):

        global enable_infra
        enable_infra = True
        self.current_infra = datos


    def main(self):

        try:
            self.rojo = bridge.imgmsg_to_cv2(self.current_infra,desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        print (self.rojo.shape, "ACAAAA")

        blob = cv2.dnn.blobFromImage(self.rojo)
        self.net.setInput(blob)
        detections = self.net.forward()


        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
            if confidence < 0.5:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

if __name__ == '__main__':

    rospy.init_node('Prueba',anonymous=True)
    model = Modelo()
    rospy.loginfo('Node started')


    while not rospy.is_shutdown():

        if enable_infra is True:
            model.main()
        else:
            print("Grabacion detenida")
    rospy.spin()
