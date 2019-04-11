#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import datetime
from numpy import array
import argparse
import imutils
import time
import dlib
import cv2
import math
import numpy
from threading import Timer
import matplotlib.pyplot as plt
import jade
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
from scipy.signal import find_peaks
from scipy.spatial import distance as dist
from scipy.signal import savgol_filter as sg
from scipy.signal import detrend as detrend
from geometry_msgs.msg import Quaternion , PoseStamped
import tf
import threading
import sys
import os
from stabilizer import Stabilizer

current_image = Image()
current_depth = Image()
bridge = CvBridge()
enable_img = False
enable_infra = False
# plt.ion()


class Camera(object):
    def __init__(self):

        #Inicializacion de variables
        #rospy.Subscriber("/camera/color/image_raw", Image, self.callback_rgb, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw" , Image , self.callback_depth,queue_size=1)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image,  self.callback_infra,queue_size=1)
        self.data_buffernose_rs=[]
        self.data_buffernose_gs=[]
        self.data_buffernose_bs=[]
        self.data_bufferforehead_rs=[]
        self.data_bufferforehead_gs=[]
        self.data_bufferforehead_bs=[]
        self.data_buffernose_gray = []
        self.data_bufferforehead_gray = []
        self.matrix_v = []
        self.nose_ica_g=[]
        self.fore_ica_g=[]
        self.time_v=[]
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'shape_predictor_68_face_landmarks.dat'))
        self.model68 = (os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'model.txt'))
        self.bpm_a = 0
        self.window_hr=0
        self.pulso_pantalla = 0
        self.pulso_guardado = 0
        self.pulso_adquirido = 0
        self.cont = 0
        self.cont2 = 0
        self.cont_yawn = 0
        self.flag = 0
        self.flag_300 = 0
        self.eye_thresh = 0.3
        self.mouth_thresh = 0.64
        self.num_frames = 1
        self.blink_counter = 0
        self.total = 0
        self.eyes_open = 0.0
        self.eyes_closed = 0.0
        self.perclos = 0.0
        self.mouth_status = False
        #self.current_depth = []
        #cv2.namedWindow('frame')
        #self.img_publisher = rospy.Publisher("topico_imagen", Image)

        self.quaternion = rospy.Publisher("quaternion", PoseStamped  , queue_size = 1)
        self.q = PoseStamped()
        self.q.header.frame_id = "map"

        """Parameters for head pose estimation"""

        self.model_points_68 = self._get_full_model_points()
        img_size=(480, 640)
        self.size = img_size
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")
        self.dist_coeefs = np.zeros((4, 1))
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

        """
        Variables parte grafica
        """
        self.blinks = [0]*40
        self.blinks_sg = [0]*40
        self.d_blinks = [0]*39
        self.mirar = [0]*1
        self.data_buffernose_gray = [0]*200
        self.data_bufferforehead_gray = [0]*200
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_ylim(40 , 100)
        self.line,  = self.ax1.plot(range(200),self.data_buffernose_gray, 'r-')
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_ylim(40 , 100)
        self.line2, = self.ax2.plot(range(200),self.data_bufferforehead_gray,'b-')
        thismanager = plt.get_current_fig_manager()
        thismanager.window.wm_geometry("+700+1")

    def callback_rgb(self,datos):

        #Callback for acquiring rgb data from the realsense
        #global enable_img
        #enable_img = True
        self.current_image = datos
        # print('callback')

    def _get_full_model_points(self, filename='model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(self.model68) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1



        return model_points

    def solve_pose_by_68_points(self, image_points):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        image_points = np.float32(image_points)

        # if self.r_vec is None:
        #     (_, rotation_vector, translation_vector) = cv2.solvePnP(
        #         self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs)
        #     self.r_vec = rotation_vector
        #     self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def callback_depth(self,datos):

        # print("entre al calback depth")

        #Callback for acquiring depth data from the realsense
        global enable_img
        enable_img = True
        self.current_depth = datos

    def callback_infra(self,datos):

        #Callback for acquiring depth data from the realsense
        # print("entre al callback infra")
        global enable_infra
        enable_infra = True
        self.current_infra = datos

    def detect_near_face(self,rects):

        dist = []
        for rect in rects:
            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)
            vector_b = numpy.array([shape[27:35]])
            (x_rr,y_rr,w_rr,h_rr)=cv2.boundingRect(vector_b)
            roi_depth = self.depth[y_rr : y_rr + h_rr , x_rr : x_rr + w_rr]
            #roi_depth = self.depth[rect.top():rect.bottom(), rect.left():rect.right()]
            mean_distance = cv2.mean(roi_depth)
            dist.append(mean_distance[0])

        index = numpy.argmin(dist)
        rectss = rects.pop(index)

        return rectss

    def add_buffer_data_matrix(self):

        self.matrix_v.append(self.green_vector)

    def close_camera(self):

        self.video_capture.release()

    def visualizacion(self,frame):

        cv2.imshow("Frame",frame)
        cv2.waitKey(1)

    def visualizacion_ros(self, img):

        image_out = Image()
        try:
            image_out = bridge.cv2_to_imgmsg(img, 'bgr8')
        except CvBridgeError as e:
            print(e)
        image_out.header = current_image.header
        self.img_publisher.publish(image_out)

    # def face_orientation(self,size,landmarks):
    #
    #     image_points = numpy.array([
    #                         (landmarks[33][0],landmarks[33][1]),     # Nose
    #                         (landmarks[8][0],landmarks[8][1]),   # Chin
    #                         (landmarks[36][0],landmarks[36][1]),     # Left eye corner
    #                         (landmarks[45][0],landmarks[45][1]),     # Right eye  corner
    #                         (landmarks[48][0],landmarks[48][1]),     # Left corner Mouth
    #                         (landmarks[54][0],landmarks[54][1])      # Right corner Mouth
    #                         ], dtype= "double")
    #
    #                         #Anthopological values
    #     model_points = numpy.array([
    #                         (0.0, 0.0, 0.0),             # Nose tip
    #                         (0.0, -330.0, -65.0),        # Chin
    #                         (-165.0, 170.0, -135.0),     # Left eye left corner
    #                         (165.0, 170.0, -135.0),      # Right eye right corne
    #                         (-150.0, -150.0, -125.0),    # Left Mouth corner
    #                         (150.0, -150.0, -125.0)      # Right mouth corner
    #                         ])
    #
    #     #Values of the camera matrix
    #     center = (size[1]/2, size[0]/2)
    #     focal_length = center[0] / numpy.tan(60/2 * numpy.pi / 180)
    #     camera_matrix = numpy.array(
    #                  [[focal_length, 0, center[0]],
    #                  [0, focal_length, center[1]],
    #                  [0, 0, 1]], dtype = "double"
    #                  )
    #
    #     dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
    #     (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #
    #     axis = numpy.float32([[500,0,0],
    #                   [0,500,0],
    #                   [0,0,500]])
    #
    #     imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    #     modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    #     rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    #     proj_matrix = numpy.hstack((rvec_matrix, translation_vector))
    #     eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    #     pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    #     pitch = math.degrees(math.asin(math.sin(pitch)))
    #     yaw = math.degrees(math.asin(math.sin(yaw)))
    #     roll = math.degrees(math.asin(math.sin(roll)))
    #
    #     return str(int(pitch)) , str(int(yaw)) , str(int(roll))

    def  eye_ratio(self,eye):

        A = dist.euclidean(eye[1],eye[5])
        B = dist.euclidean(eye[2],eye[4])
        C = dist.euclidean(eye[0],eye[3])
        eye_ratio = (A+B)#/(C * 2.0)
        # print(eye_ratio)

        return eye_ratio

    def mouth_ratio(self,mou):
        #Horizontal
        a   = dist.euclidean(mou[0], mou[6])
        #Vertical
        b  = dist.euclidean(mou[3], mou[9])
        c  = dist.euclidean(mou[2], mou[10])
        d  = dist.euclidean(mou[4], mou[8])
        e   = (b+c+d)/3

        mouth_ratio = e/a

        return mouth_ratio

    def get_pulse(self,ad_hr):

        # T + 1 pulse adquiction
        self.pulso_guardado = self.pulso_adquirido
        # T pulse adquiction
        self.pulso_adquirido = ad_hr
        comparation = numpy.absolute(self.pulso_guardado - self.pulso_adquirido)
        if comparation < 1:
            #Displayed pulse
            self.pulso_pantalla = (self.pulso_guardado + self.pulso_adquirido)/2

        return self.pulso_pantalla

    def add_buffer_data(self):

        self.data_buffernose_gray.pop(0)
        self.data_bufferforehead_gray.pop(0)
        #self.data_buffernose_rs.append(self.m_nr[0])
        #self.data_buffernose_gs.append(self.m_ng[0])
        #self.data_buffernose_bs.append(self.m_nb[0])
        self.data_buffernose_gray.append(self.nose_gray[0])
        #self.data_bufferforehead_rs.append(self.m_fr[0])
        #self.data_bufferforehead_gs.append(self.m_fg[0])
        #self.data_bufferforehead_bs.append(self.m_fb[0])
        self.data_bufferforehead_gray.append(self.forehead_gray[0])
        self.cont += 1

    def add_buffer_data_2(self):

        #self.data_buffernose_rs[self.cont]=(self.m_nr[0])
        #self.data_buffernose_gs[self.cont]=(self.m_ng[0])
        #self.data_buffernose_bs[self.cont]=(self.m_nb[0])
        # self.data_buffernose_gray[self.cont]=(self.nose_gray[0])
        #self.data_bufferforehead_rs[self.cont]=(self.m_fr[0])
        #self.data_bufferforehead_gs[self.cont]=(self.m_fg[0])
        #self.data_bufferforehead_bs[self.cont]=(self.m_fb[0])
        # self.data_bufferforehead_gray[self.cont]=(self.forehead_gray[0])
        self.data_buffernose_gray.pop(0)
        self.data_buffernose_gray.append(self.nose_gray[0])
        self.data_bufferforehead_gray.pop(0)
        self.data_bufferforehead_gray.append(self.forehead_gray[0])
        #print("ENTRE ACA")
        self.cont += 1
        self.cont2 += 1

        if self.cont > 199:
            self.cont = 0
        else:
            pass
        if self.cont2 == 20:
            self.cont2 = 0
            self.normalizacion()
            print("ENTREEEEEEEEEEEE")
        else:
            pass

    def bandpass_filter(self, data, lowcut, highcut, fs):

        order = 5.0
        nyq = 0.5 * fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = butter(order,[low,high],btype='band')
        y = lfilter(b,a,data)
        return y

    def normalizacion(self):
        #print(numpy.shape(self.time_v))
        self.flag_300 =1
        #self.n_frame=len(self.data_buffernose_gs)
        self.n_frame = len(self.data_buffernose_gray)
        self.time = numpy.mean(self.time_v)
        self.fs = 1.0/self.time

        #Calculando la fs cuando se toma todo el tiempo de adquicision
        #self.time = (time.time()-self.current)
        #print(self.n_frame)
        #print(self.time)
        #self.fs= self.n_frame/self.time

        self.xf = numpy.linspace(0.0,(self.fs/2),self.n_frame/2)

        # Normalizacion datos
        #n_datanose_rs = (self.data_buffernose_rs-numpy.mean(self.data_buffernose_rs))/numpy.std(self.data_buffernose_rs)
        #n_datanose_gs = (self.data_buffernose_gs-numpy.mean(self.data_buffernose_gs))/numpy.std(self.data_buffernose_gs)
        #n_datanose_bs = (self.data_buffernose_bs-numpy.mean(self.data_buffernose_bs))/numpy.std(self.data_buffernose_bs)
        n_datanose_gray = (self.data_buffernose_gray - numpy.mean(self.data_buffernose_gray))/numpy.std(self.data_buffernose_gray)
        #n_datafore_rs = (self.data_bufferforehead_rs-numpy.mean(self.data_bufferforehead_rs))/numpy.std(self.data_bufferforehead_rs)
        #n_datafore_gs = (self.data_bufferforehead_gs-numpy.mean(self.data_bufferforehead_gs))/numpy.std(self.data_bufferforehead_gs)
        #n_datafore_bs = (self.data_bufferforehead_bs-numpy.mean(self.data_bufferforehead_bs))/numpy.std(self.data_bufferforehead_bs)
        n_datafore_gray = (self.data_bufferforehead_gray - numpy.mean(self.data_bufferforehead_gray))/numpy.std(self.data_bufferforehead_gray)


        # Creating matrix for ICA
        #ica_fore =numpy.zeros((3,len(n_datafore_rs)))
        #ica_nose =numpy.zeros((3,len(n_datanose_rs)))
        #ica_nose[0,:]= n_datanose_rs
        #ica_nose[1,:]= n_datanose_gs
        #ica_nose[2,:]= n_datanose_bs
        ica_nose = n_datanose_gray
        #ica_fore[0,:]= n_datafore_rs
        #ica_fore[1,:]= n_datafore_gs
        #ica_fore[2,:]= n_datafore_bs
        ica_fore = n_datafore_gray

        self.ica_both = numpy.zeros((2,len(n_datanose_gray)))
        self.ica_both[0,:] = ica_nose
        self.ica_both[1,:] = ica_fore
        #self.matrix_v = array(self.matrix_v)
        #ica_vector = jade.main(self.matrix_v,1)
        thread_a  = threading.Thread(target = self.ICA)
        thread_a.start()
        #thread_a.join()

    def ICA(self):

        # Applying ICA
        nose_ica = jade.main(self.ica_both)
        #fore_ica = jade.main(nose)

        # Transpose of ICA result
        nose_ica=nose_ica.T
        #fore_ica=fore_ica.T
        #self.nose_ica_g = nose_ica[1,:]
        #self.fore_ica_g = fore_ica[1,:]

        #nose_green = nose_ica[1,:]
        nose_green = nose_ica[0,:]
        nose_green = numpy.ravel(nose_green)
        nose_green = numpy.hamming(self.n_frame) * nose_green

        #fore_green = fore_ica[1,:]
        #fore_green = numpy.ravel(fore_green)
        #fore_green = numpy.hamming(self.n_frame) * fore_green

        #Bandpass filter on the green channel on nose and forehead
        nose_green= self.bandpass_filter(nose_green,0.95,2.0,self.fs)
        #fore_ica[1,:]= self.bandpass_filter(fore_ica[1,:],0.75,2.0,self.fs)
        fore_ica = self.bandpass_filter(nose_ica[1,:],0.95,2.0,self.fs)

        #Fourier transform
        #r_nosefft= numpy.absolute(numpy.fft.fft(nose_ica[0,:], norm="ortho"))
        g_nosefft= numpy.absolute(numpy.fft.fft(nose_green, norm="ortho"))
        f_forefft= numpy.absolute(numpy.fft.fft(fore_ica, norm="ortho"))
        #b_nosefft= numpy.absolute(numpy.fft.fft(nose_ica[2,:], norm="ortho"))
        #r_forefft= numpy.absolute(numpy.fft.fft(fore_ica[0,:], norm="ortho"))
        #g_forefft= numpy.absolute(numpy.fft.fft(fore_ica, norm="ortho"))
        #b_forefft= numpy.absolute(numpy.fft.fft(fore_ica[2,:], norm="ortho"))
        g_nosefft= numpy.ravel(g_nosefft)
        g_forefft= numpy.ravel(f_forefft)

        #Only takes de half of the data
        y_plot_nose = g_nosefft[:self.n_frame//2]
        y_plot_fore = g_forefft[:self.n_frame//2]

        self.index_nose = numpy.argmax(y_plot_nose)
        self.index_fore = numpy.argmax(y_plot_fore)

        self.bpm_a = self.xf[self.index_nose] * 60
        self.bpm_b = self.xf[self.index_fore] * 60

        #Gets the pulse
        self.window_hr = self.get_pulse(self.bpm_a)
        print(self.bpm_a,"BPM componente 1")
        print(self.bpm_b,"BPM componente 2")
        #thread_b  = threading.Thread(target = self.plotting, args =(self.nose_ica_g,self.fore_ica_g))
        #thread_b.start()
        #Plot fourier
        #plt.clf()
        #plt.plot(self.xf,y_plot_fore,'b')
        #plt.plot(self.xf,y_plot_nose,'r')
        #plt.plot(self.xf[(self.xf > 0.70)],y_plot_nose[(self.xf > 0.70)],'r',label="Nose")
        #plt.plot(self.xf[(self.xf > 0.70)],y_plot_fore[(self.xf > 0.70)],'b',label="Forehead")
        #plt.title("Red-Nose, Blue-Forehead")
        #plt.show()

    def plotting(self,x1):

        #plt.clf()
        #x1=numpy.ravel(x1)
        #x2=numpy.ravel(x2)
        #plt.plot(x1)
        #plt.plot(x2)
        #plt.show()

        # if len(self.data_buffer_eyeblink) > 40:
        #self.blinks.pop(0)
        #self.blinks.append(x1)

        # self.blinks_sg = sg(self.blinks, 5, 2)
        # self.blinks_sg = self.blinks_sg - np.mean(self.blinks_sg)
        # self.blinks_sg= self.bandpass_filter(self.blinks_sg,0.1,10.0,self.fs)
        # self.blinks_sg = detrend(self.blinks_sg)
        # self.blinks_sg = -self.blinks_sg
        # print(self.blinks)

        # [p ,l] = find_peaks((-1 * self.blinks))
        # self.d_blinks = numpy.diff(self.blinks)

        # peaks , _ = find_peaks(self.blinks)
        #print peaks.shape
        # self.mirar.pop(0)
        # self.mirar.append(self.d_blinks[38])
        #print(self.mirar)
        #
        # peaks = find_peaks(self.blinks)
        # list = [0]*40
        # for i in peaks:
        #     pass
        #
        #
        self.line.set_ydata(self.data_buffernose_gray)
        self.line2.set_ydata(self.data_bufferforehead_gray)

    def main(self):

        #try:
        #    self.frames = bridge.imgmsg_to_cv2(self.current_image,"bgr8")
        #except CvBridgeError as e:
        #    print(e)

        try:
            self.depth = bridge.imgmsg_to_cv2(self.current_depth,desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        try:
            self.rojo = bridge.imgmsg_to_cv2(self.current_infra,desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        # self.frames = frame
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.prev_mouth_status = self.mouth_status
        self.current = time.time()
        #self.frames = imutils.resize(self.frames, width=400)
        #self.depth = imutils.resize(self.depth, width=400)
        #self.gray = #cv2.cvtColor(self.frames, cv2.COLOR_BGR2GRAY)
        self.gray = self.rojo
        self.gray = cv2.equalizeHist(self.gray) ##Histogram equalization
        rects = self.detector(self.gray, 0)
        size = self.rojo.shape
        #print(size)


        #print(len(rects))

        if not rects:

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (15,20)
            fontScale = 0.65
            fontColor = (255,0,0)
            lineType  = 1
            cv2.putText(self.rojo,'NO DETECTION!', bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            if self.flag_300 ==0:
                self.data_buffernose_rs=[]
                self.data_buffernose_gs=[]
                self.data_buffernose_bs=[]
                self.data_bufferforehead_rs=[]
                self.data_bufferforehead_gs=[]
                self.data_bufferforehead_bs=[]
                self.data_buffernose_gray = []
                self.data_bufferforehead_gray = []
                self.cont=0
            else:
                pass
            self.time_v = []
            self.visualizacion(self.rojo)
            #self.visualizacion_ros(self.frames)
            self.flag = 1

        else:

            if len(rects) > 1:
                rect = self.detect_near_face(rects)
            elif len(rects) == 1:
                rect = rects.pop()
            else:
                return

            #vector_b = numpy.array([shape[27:35]])
            #(x_rr,y_rr,w_rr,h_rr)=cv2.boundingRect(vector_b)
            #roi_depth = self.depth[y_rr : y_rr + h_rr , x_rr : x_rr + w_rr]

            #roi_depth = self.depth[rect.top():rect.bottom(), rect.left():rect.right()]
            #mean_distance = cv2.mean(roi_depth)
            #print(mean_distance[0])
            #self.visualizacion(roi_depth)

            self.flag = 0

            top  = rect.top()
            bottom = rect.bottom() +20
            left = rect.left()
            right = rect.right()

            recta = dlib.rectangle(left,top,right,bottom)

            shape = self.predictor(self.gray, recta)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
            for (x, y) in shape[27:35]:
                cv2.circle(self.rojo, (x, y), 1, (0, 255, 0), -1)
            #Get the ROI of the nose
            vector_a=numpy.array([shape[27:35]])
            (x_r,y_r,w_r,h_r)=cv2.boundingRect(vector_a)
            roi_visualizacion_nose = self.rojo[y_r:y_r+h_r,x_r:x_r+w_r]
            #roi_visualizacion_nose = imutils.resize(roi_visualizacion_nose, width=100)

            for (x, y) in shape[48:60]:
                cv2.circle(self.rojo, (x, y), 1, (0, 255, 0), -1)

            # draw over the (x,y) coordinates of the rigth and left eyebrow
            (x_ci,y_ci) = shape[20]
            cv2.circle(self.rojo, (x_ci,y_ci), 1, (0,255,0),-1)
            (x_cd,y_cd) = shape[23]
            cv2.circle(self.rojo, (x_cd,y_cd), 1, (0,255,0),-1)

            # draw the ROI of the forehead
            cv2.circle(self.rojo,(x_ci,y_ci-10), 1,(255,0,0),-1)
            cv2.circle(self.rojo,(x_cd,y_cd-10),1, (255,0,0),-1)
            cv2.rectangle(self.rojo,(x_ci,y_ci-25),(x_cd,y_cd-10),(255,0,0),1)
            roi_visualizacion_forehead = self.rojo[y_ci-25:y_cd-10, x_ci:x_cd]

            # get the area of the mouth
            #vector_b=numpy.array([shape[48:59]])
            #(x_rr,y_rr,w_rr,h_rr) = cv2.boundingRect(vector_b)
            #area = w_rr * h_rr
        #    print(area,"Area")
    #        print(h_rr , "Height")




            # extract each channel from the nose ROI
            #nose_r = roi_visualizacion_nose[:,:,0]
            #nose_g = roi_visualizacion_nose[:,:,1]
            #nose_b = roi_visualizacion_nose[:,:,2]
            nose_gray = roi_visualizacion_nose

            # extract each channel from the forehead ROI
            #forehead_r = roi_visualizacion_forehead[:,:,0]
            #forehead_g = roi_visualizacion_forehead[:,:,1]
            #forehead_b = roi_visualizacion_forehead[:,:,2]
            forehead_gray = roi_visualizacion_forehead

            #Median filter of each channel
            #nose_r = cv2.medianBlur(nose_r,3)
            #nose_g = cv2.medianBlur(nose_g,3)
            #nose_b = cv2.medianBlur(nose_b,3)
            nose_gray = cv2.medianBlur(nose_gray,3)
            #forehead_r = cv2.medianBlur(forehead_r,3)
            #forehead_g = cv2.medianBlur(forehead_g,3)
            #forehead_b = cv2.medianBlur(forehead_b,3)
            forehead_gray = cv2.medianBlur(forehead_gray,3)

            # Make the green matrix a vector
            #print(nose_g.shape,"MATRIIIIIIIZ")
            #self.green_vector = numpy.reshape(nose_g, (nose_r.shape[0]*nose_r.shape[1],1))
            #self.green_vector = self.green_vector[:1]
            #print(self.green_vector.shape,"VECTOOOOOOOOOOOOOOOOOOOOOOR")

            # Spatial average of each channel
            #self.m_nr = cv2.mean(nose_r)
            #self.m_ng = cv2.mean(nose_g)
            #self.m_nb = cv2.mean(nose_b)
            self.nose_gray = cv2.mean(nose_gray)
            #self.m_fr = cv2.mean(forehead_r)
            #self.m_fg = cv2.mean(forehead_g)
            #self.m_fb = cv2.mean(forehead_b)
            self.forehead_gray = cv2.mean(forehead_gray)

            #Extract the left and rigth coordinates from left and rigth eye,then calculate eye ratio
            #And extract the coordinates of the mouth
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = self.eye_ratio(leftEye)
            rightEAR = self.eye_ratio(rightEye)
            mouAR = self.mouth_ratio(mouth)

            #print(mouAR,"MOUTH AREA")
            # Compute each eye ratio
            eye_r = (leftEAR + rightEAR) / 2.0
            # eye_r = eye_r * eye_r

            ####print(eye_r)
            # print(eye_r)
            # self.plotting(eye_r)
            #count the blinks
            if eye_r < 8.3:
                self.blink_counter += 1
                self.eyes_closed += 1

                if self.blink_counter > 50:
                    print("SE ESTA QUEDANDO DORMIDO")

            else:
                self.eyes_open += 1
                if self.blink_counter >= self.num_frames:
                    self.total += 1
                self.blink_counter = 0

            if (self.eyes_open + self.eyes_closed) > 545:
                self.perclos = (self.eyes_closed/(self.eyes_open+self.eyes_closed))*100
                print(self.eyes_closed,"Contador ojos cerrados")
                print(self.eyes_open,"Contador ojos abiertos")
                print(self.perclos,"PERCLOS")
                if self.perclos > 40.0:
                    print("ALERTA MICROSUENO")
                self.eyes_open = 0.0
                self.eyes_closed = 0.0

            #print(mouAR)

            if mouAR > self.mouth_thresh:
                self.mouth_status = True

            else:
                self.mouth_status = False

            if self.prev_mouth_status == True and self.mouth_status == False:
                self.cont_yawn +=1

            #Gets the head pose
            #angle_pitch , yaw , roll = self.face_orientation(size,shape)

            pose = self.solve_pose_by_68_points(shape)
            stabile_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, self.pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
            stabile_pose = np.reshape(stabile_pose, (-1, 3))
            rotation_mat, _ = cv2.Rodrigues(stabile_pose[0])
            pose_mat = cv2.hconcat((rotation_mat, stabile_pose[1]))
            _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

            self.q.pose.orientation.x , self.q.pose.orientation.y , self.q.pose.orientation.z , self.q.pose.orientation.w = tf.transformations.quaternion_from_euler(euler_angle[1],
            euler_angle[0], 0)
            self.quaternion.publish(self.q)

            #Put the information into the frame
            cv2.putText(self.rojo,'Num. Parpadeos:{}' .format(self.total), (15,20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 0, 0),1)
            cv2.putText(self.rojo,'Num. Bostezos:{}' .format(self.cont_yawn), (15,40),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 0, 0),1)
            #cv2.putText(self.frames,'ER: {}' .format(er), (300,15),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 0, 255),1)
            cv2.putText(self.rojo,'HR:{}' .format(self.window_hr), (15,460),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 255),1)
            cv2.putText(self.rojo,'PITCH:{}' .format(euler_angle[0]), (490,420),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
            cv2.putText(self.rojo,'YAW:{}' .format(euler_angle[1]), (490,440),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)
            cv2.putText(self.rojo,'ROLL:{}' .format(euler_angle[2]), (490,460),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255),1)

            #Append the data on the buffer and display the video


            #self.visualizacion_ros(self.frames)
            self.visualizacion(self.rojo)
            self.actual_time = (time.time()-self.current)
            self.time_v.append(self.actual_time)
            # print(self.actual_time)

            if self.flag_300 == 0:
                self.add_buffer_data()
                if self.cont == 200:
                    self.normalizacion()
                    self.cont = 0
                else:
                    pass
            else:
                self.add_buffer_data_2()
            # self.plotting(None)
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()

def main():

    rospy.init_node('listener',anonymous=True)
    print("ACA")
    cam = Camera()

    rospy.loginfo('Node started')

    while not rospy.is_shutdown():

        if enable_img is True and enable_infra is True:
            # print("entre")
            #print("Entre")
            cam.main()
    #rospy.sleep(0.01)
    rospy.spin()
#    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
