#!/usr/bin/env python
import rospy
import numpy as np
import dlib
import cv2
from face_recognition.msg import Faces
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class FaceDetection(object):
	def __init__(self):
		"""Parameters"""
		self.rotations = rospy.get_param('~rotation_cycles', 3)
		self.rgb_camera = rospy.get_param('~rgb_camera', True)
		self.depth_camera = rospy.get_param('~depth_camera', False)
		self.ir_camera = rospy.get_param('~ir_camera', False)
		rgb_image_encoding = rospy.get_param('~rgb_image_encoding', "bgr8") #rgb8
		ir_image_encoding = rospy.get_param('~ir_image_encoding', "mono16") #mono8
		self.depth_image_encoding = rospy.get_param('~depth_image_encoding', "mono16") #mono8
		self.multiple_detection = rospy.get_param('~multiple_detection', False)
		self.show_detection = rospy.get_param('~show_detection', False)
		"""Subscribers"""
		if self.rgb_camera:
			self.sub_bgr_image = rospy.Subscriber("color_image",Image,self.callback_image)
			self.image_encoding = rgb_image_encoding
		elif self.ir_camera:
			self.sub_ir_image = rospy.Subscriber("ir_image",Image,self.callback_image)
			self.image_encoding = ir_image_encoding

		if self.depth_camera:
			self.sub_bgr_image = rospy.Subscriber("depth_image",Image,self.callback_depth_image)
		"""Publishers"""
		self.pub_face_image = rospy.Publisher("faces_images",Faces,queue_size = 10)
		if self.show_detection:
			self.pub_detections = rospy.Publisher("detected_faces",Image,queue_size = 10)
		"""Node Configuration"""
		self.bridge = CvBridge()
		self.depth_image = None
		self.image = None
		self.face_detector = dlib.get_frontal_face_detector()
		self.image_shape = None #[h,w]

	def callback_image(self,msg):
		try:
			image = self.bridge.imgmsg_to_cv2(msg, self.image_encoding)
		except CvBridgeError as e:
			print(e)
		self.image = np.rot90(image, k=self.rotations)

		if self.image_shape is None:
			self.image_shape = self.image.shape

		detected_faces = Faces()
		detected_faces.faces_images = []
		detected_faces.faces_depth_images = []
		rects = self.face_detector(image, 0)

		if len(rects) == 0:
			people_detected = False
		else:
			people_detected = True

		if not(self.multiple_detection):
			closest_rect = None
		if self.depth_camera:
			while(self.depth_image is None):
				pass
		for rect in rects:
			if self.multiple_detection:
				crop_bgr_img,crop_depth_img = self.crop_images(image,rect)
				try:
					detected_faces.faces_images.append(self.bridge.cv2_to_imgmsg(crop_bgr_img, self.image_encoding))
					if self.depth_camera:
						detected_faces.faces_depth_images.append(self.bridge.cv2_to_imgmsg(crop_depth_img,self.depth_image_encoding))
				except CvBridgeError as e:
					print(e)
				if self.show_detection:
					cv2.rectangle(image, (rect.left(),rect.top()), (rect.right(),rect.bottom()), (0,255,0),2)
			else:
				if closest_rect is None:
					closest_rect = rect
				else:
					if self.depth_camera:
						roi_depth_img = self.depth_image[max(0, rect.top()): min(rect.bottom(), self.image_shape[0]),
														  max(0, rect.left()): min(rect.right(), self.image_shape[1])]
						closest_rect_roi_depth_img = self.depth_image[max(0, rect.top()): min(rect.bottom(), self.image_shape[0]),
																	  max(0, rect.left()): min(rect.right(), self.image_shape[1])]
						if cv2.mean(roi_depth_img) < cv2.mean(closest_rect_roi_depth_img):
							closest_rect = rect
					else:
						if rect.area() > closest_rect.area():
							closest_rect = rect

		if people_detected:
			if not(self.multiple_detection):
				crop_bgr_img,crop_depth_img = self.crop_images(image,closest_rect)
				try:
					detected_faces.faces_images.append(self.bridge.cv2_to_imgmsg(crop_bgr_img, self.image_encoding))
					if self.depth_camera:
						detected_faces.faces_depth_images.append(self.bridge.cv2_to_imgmsg(crop_depth_img,self.depth_image_encoding))
				except CvBridgeError as e:
					print(e)

				if self.show_detection:
					cv2.rectangle(image, (closest_rect.left(),closest_rect.top()), (closest_rect.right(),closest_rect.bottom()), (0,255,0),2)

				self.pub_face_image.publish(detected_faces)

			if self.show_detection:
				self.pub_detections.publish(self.bridge.cv2_to_imgmsg(image,self.image_encoding))
		self.depth_image = None

	def callback_depth_image(self,msg):
		if self.depth_image is None:
			try:
				image = self.bridge.imgmsg_to_cv2(msg, self.depth_image_encoding)
			except CvBridgeError as e:
				print(e)
			self.depth_image = np.rot90(image, k=self.rotations)

	def crop_images(self,image,rect):
		crop_bgr_img = image[max(0, rect.top()): min(rect.bottom(), self.image_shape[0]),
							 max(0, rect.left()): min(rect.right(), self.image_shape[1])]
		if self.depth_camera:
			crop_depth_img = self.depth_image[max(0, rect.top()): min(rect.bottom(), self.image_shape[0]),
											  max(0, rect.left()): min(rect.right(), self.image_shape[1])]
		else:
			crop_depth_img = None
		return crop_bgr_img,crop_depth_img

if __name__ == '__main__':
	try:
		rospy.init_node("face_detection", anonymous = True)
		face_detection = FaceDetection()
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()
