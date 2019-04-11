#!/usr/bin/env python
import rospy
import numpy as np
import dlib
import cv2
from face_recognition.msg import Faces, SingleLandmarks, MultipleLandmarks
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from time import time


class FeatureExtraction(object):
    def __init__(self):
        """Parameters"""
        self.depth_camera = rospy.get_param('~depth_camera', False)
        self.image_encoding = rospy.get_param('~image_encoding', "bgr8") #passthrougth
        self.depth_image_encoding = rospy.get_param('~depth_image_encoding', "passthrougth") #mono8
        self.extraction_method = rospy.get_param('~extraction_method', "landmarks") #mono8
        self.show_features = rospy.get_param('~show_features', False)
        if self.extraction_method == "landmarks":
            self.get_features = self.landmarks_based_extraction
            self.msg_type = MultipleLandmarks
        elif self.extraction_method == "pca":
            self.get_features = self.pca_based_extraction
            self.msg_type = Image
        else:
            rospy.logerr("Invalid feature extraction method")
            exit()
        """Subscribers"""
        self.sub_faces = rospy.Subscriber("faces_images",Faces,self.callback_faces)
        """Publishers"""
        self.pub_faces_features = rospy.Publisher("faces_features",self.msg_type,queue_size = 10)
        rospy.loginfo("Show:{}".format(self.show_features) )
        if self.show_features:
            self.pub_faces_features_img = rospy.Publisher("faces_features/img",Image,queue_size = 10)
        """Node Configuration"""
        self.bridge = CvBridge()
        self.faces_imgs = None
        self.faces_depth_imgs = None
        if self.extraction_method == "landmarks":
            self.face_predictor = dlib.shape_predictor(PREDICTOR_PATH)
            self.face_descriptor = dlib.face_recognition_model_v1(DESCRIPTOR_PATH)

        self.main()

    def callback_faces(self,msg):
        if self.faces_imgs is None:
            try:
                # self.faces_imgs = [cv2.cvtColor(self.bridge.imgmsg_to_cv2(img, self.image_encoding),cv2.COLOR_BGR2GRAY) for img in msg.faces_images]
                self.faces_imgs = [self.bridge.imgmsg_to_cv2(img, self.image_encoding) for img in msg.faces_images]
                self.faces_depth_imgs = [self.bridge.imgmsg_to_cv2(img, self.depth_image_encoding) for img in msg.faces_depth_images]
            except CvBridgeError as e:
                print(e)

    def landmarks_based_extraction(self):
        features = self.msg_type()
        features.data = []
        landmarks = SingleLandmarks()
        # if self.show_features:
        #     faces = np.array([])

        for face in self.faces_imgs:
            t = time()
            img_shape = face.shape
            print(img_shape)

            shape = self.face_predictor(face, dlib.rectangle(0,0,img_shape[1],img_shape[0]))

            face_descriptor = self.face_descriptor.compute_face_descriptor(face, shape)
            # print(type(face_descriptor),type(face_descriptor[1]))
            shape = np.array([[int(shape.part(i).x),int(shape.part(i).y)] for i in range(68)])
            landmarks.x = shape[:,0]
            landmarks.y = shape[:,1]
            features.data.append(landmarks)
            print(time()-t)


        return features

    def pca_based_extraction(self):
        pass
        # if self.faces_imgs is None:
        #     try:
        #         image = self.bridge.imgmsg_to_cv2(msg, self.depth_image_encoding)
        #     except CvBridgeError as e:
        #         print(e)
        #     self.depth_image = np.rot90(image, k=self.rotations)

    def main(self):
        while not rospy.is_shutdown():
            if not(self.faces_imgs is None) and (not(self.depth_camera) or not(self.faces_depth_imgs is None)):
                features = self.get_features()
                self.faces_imgs = self.faces_depth_imgs = None
                self.pub_faces_features.publish(features)

if __name__ == '__main__':
    try:
        from rospkg import RosPack
        PREDICTOR_PATH = RosPack().get_path("face_recognition")+"/include/shape_predictor_68_face_landmarks.dat"
        DESCRIPTOR_PATH = RosPack().get_path("face_recognition")+"/include/dlib_face_recognition_resnet_model_v10.dat"
        rospy.init_node("face_detection", anonymous = True)
        feature_extraction = FeatureExtraction()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
