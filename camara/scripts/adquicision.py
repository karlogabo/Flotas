#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from camara.msg import driver_info
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
import threading
import sys
import os

current_image = Image()
current_depth = Image()
bridge = CvBridge()
enable_img = False

class Camera(object):

    def __init__(self):

        rospy.Subscriber("/camera/color/image_raw", Image, self.callback_rgb, queue_size=1)
        rospy.Subscriber("/camera/depth/image_rect_raw" , Image , self.callback_depth,queue_size=1)
        self.info = rospy.Publisher("driver_info", driver_info)
        self.msg = driver_info()

        self.data_buffernose_rs =[]
        self.data_buffernose_gs =[]
        self.data_buffernose_bs = []
        self.data_bufferforehead_rs =[]
        self.data_bufferforehead_gs =[]
        self.data_bufferforehead_bs = []
        self.nose_ica_g = []
        self.fore_ica_g = []
        self.matrix_v = []
        self.time_v=[]
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'shape_predictor_68_face_landmarks.dat'))
        self.bpm_a = 0
        self.window_hr = 0
        self.pulso_pantalla = 0
        self.pulso_guardado = 0
        self.pulso_adquirido = 0
        self.cont = 0
        self.cont2 = 0
        self.cont_yawn =0
        self.flag = 0
        self.flag_300 = 0
        self.blink_counter = 0
        self.total = 0
        self.eye_thresh = 0.3
        self.mouth_thresh = 0.5
        self.num_frames = 2
        self.eyes_open =0.0
        self.eyes_closed =0.0
        self.perclos = 0.0
        self.mouth_status = False
        self.contador_detec = 0

    def close_camera(self):

        self.video_capture.release()

    def visualizacion(self,frame):

        cv2.imshow("Frame",frame)
        cv2.waitKey(1)

    def callback_rgb(self,datos):

        #Callback for acquiring rgb data from the realsense
        global enable_img
        enable_img = True
        self.current_image = datos

    def callback_depth(self,datos):

        #Callback for acquiring depth data from the realsense
        global enable_depth
        enable_depth = False
        self.current_depth = datos

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

    def face_orientation(self,size,landmarks):

        image_points = numpy.array([
                            (landmarks[33][0],landmarks[33][1]),     # Nose
                            (landmarks[8][0],landmarks[8][1]),   # Chin
                            (landmarks[36][0],landmarks[36][1]),     # Left eye corner
                            (landmarks[45][0],landmarks[45][1]),     # Right eye  corner
                            (landmarks[48][0],landmarks[48][1]),     # Left corner Mouth
                            (landmarks[54][0],landmarks[54][1])      # Right corner Mouth
                            ], dtype= "double")

                            #Anthopological values
        model_points = numpy.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

        #Values of the camera matrix
        center = (size[1]/2, size[0]/2)
        focal_length = center[0] / numpy.tan(60/2 * numpy.pi / 180)
        camera_matrix = numpy.array(
                     [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype = "double"
                     )

        dist_coeffs = numpy.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        axis = numpy.float32([[500,0,0],
                      [0,500,0],
                      [0,0,500]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = numpy.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        roll = math.degrees(math.asin(math.sin(roll)))

        return (int(pitch)) , (int(yaw)) , (int(roll))

    def  eye_ratio(self,eye):

        A = dist.euclidean(eye[1],eye[5])
        B = dist.euclidean(eye[2],eye[4])
        C = dist.euclidean(eye[0],eye[3])
        eye_ratio = (A+B)/(C * 2.0)

        return eye_ratio

    def mouth_ratio(self,mou):
        #Horizontal
        a   = dist.euclidean(mou[12], mou[16])
        #Vertical
        b  = dist.euclidean(mou[13], mou[19])
        c  = dist.euclidean(mou[14], mou[18])
        d  = dist.euclidean(mou[15], mou[17])
        e   = (b+c+d)
        mouth_ratio = e/(a*2.0)

        return mouth_ratio

    def add_buffer_data(self):

        self.data_buffernose_rs.append(self.m_nr[0])
        self.data_buffernose_gs.append(self.m_ng[0])
        self.data_buffernose_bs.append(self.m_nb[0])
        self.data_bufferforehead_rs.append(self.m_fr[0])
        self.data_bufferforehead_gs.append(self.m_fg[0])
        self.data_bufferforehead_bs.append(self.m_fb[0])
        self.cont += 1

    def add_buffer_data_2(self):

        self.data_buffernose_rs[self.cont]=(self.m_nr[0])
        self.data_buffernose_gs[self.cont]=(self.m_ng[0])
        self.data_buffernose_bs[self.cont]=(self.m_nb[0])
        self.data_bufferforehead_rs[self.cont]=(self.m_fr[0])
        self.data_bufferforehead_gs[self.cont]=(self.m_fg[0])
        self.data_bufferforehead_bs[self.cont]=(self.m_fb[0])
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

    def normalizacion(self):
        #print(numpy.shape(self.time_v))
        self.flag_300 =1
        self.n_frame=len(self.data_buffernose_gs)
        self.time = numpy.mean(self.time_v)
        self.fs = 1.0/self.time

        #Calculando la fs cuando se toma todo el tiempo de adquicision
        #self.time = (time.time()-self.current)
        #print(self.n_frame)
        #print(self.time)
        #self.fs= self.n_frame/self.time

        self.xf = numpy.linspace(0.0,(self.fs/2),self.n_frame/2)

        # Normalizacion datos
        n_datanose_rs = (self.data_buffernose_rs-numpy.mean(self.data_buffernose_rs))/numpy.std(self.data_buffernose_rs)
        n_datanose_gs = (self.data_buffernose_gs-numpy.mean(self.data_buffernose_gs))/numpy.std(self.data_buffernose_gs)
        n_datanose_bs = (self.data_buffernose_bs-numpy.mean(self.data_buffernose_bs))/numpy.std(self.data_buffernose_bs)
        n_datafore_rs = (self.data_bufferforehead_rs-numpy.mean(self.data_bufferforehead_rs))/numpy.std(self.data_bufferforehead_rs)
        n_datafore_gs = (self.data_bufferforehead_gs-numpy.mean(self.data_bufferforehead_gs))/numpy.std(self.data_bufferforehead_gs)
        n_datafore_bs = (self.data_bufferforehead_bs-numpy.mean(self.data_bufferforehead_bs))/numpy.std(self.data_bufferforehead_bs)

        # Creating matrix for ICA
        ica_fore =numpy.zeros((3,len(n_datafore_rs)))
        ica_nose =numpy.zeros((3,len(n_datanose_rs)))
        ica_nose[0,:]= n_datanose_rs
        ica_nose[1,:]= n_datanose_gs
        ica_nose[2,:]= n_datanose_bs
        ica_fore[0,:]= n_datafore_rs
        ica_fore[1,:]= n_datafore_gs
        ica_fore[2,:]= n_datafore_bs
        #self.matrix_v = array(self.matrix_v)
        #ica_vector = jade.main(self.matrix_v,1)
        thread_a  = threading.Thread(target = self.ICA, args =(ica_fore,ica_nose))
        thread_a.start()
        #thread_a.join()

    def ICA(self, forehead, nose):

        # Applying ICA
        nose_ica = jade.main(forehead)
        fore_ica = jade.main(nose)

        # Transpose of ICA result
        nose_ica=nose_ica.T
        fore_ica=fore_ica.T
        self.nose_ica_g = nose_ica[1,:]
        self.fore_ica_g = fore_ica[1,:]

        nose_green = nose_ica[1,:]
        nose_green = numpy.ravel(nose_green)
        nose_green = numpy.hamming(self.n_frame) * nose_green

        #fore_green = fore_ica[1,:]
        #fore_green = numpy.ravel(fore_green)
        #fore_green = numpy.hamming(self.n_frame) * fore_green

        #Bandpass filter on the green channel on nose and forehead
        nose_green= self.bandpass_filter(nose_green,0.95,2.0,self.fs)
        fore_ica[1,:]= self.bandpass_filter(fore_ica[1,:],0.75,2.0,self.fs)

        #Fourier transform
        r_nosefft= numpy.absolute(numpy.fft.fft(nose_ica[0,:], norm="ortho"))
        g_nosefft= numpy.absolute(numpy.fft.fft(nose_green, norm="ortho"))
        b_nosefft= numpy.absolute(numpy.fft.fft(nose_ica[2,:], norm="ortho"))
        r_forefft= numpy.absolute(numpy.fft.fft(fore_ica[0,:], norm="ortho"))
        g_forefft= numpy.absolute(numpy.fft.fft(fore_ica[1,:], norm="ortho"))
        b_forefft= numpy.absolute(numpy.fft.fft(fore_ica[2,:], norm="ortho"))
        g_nosefft= numpy.ravel(g_nosefft)
        g_forefft= numpy.ravel(g_forefft)

        #Only takes de half of the data
        y_plot_nose = g_nosefft[:self.n_frame//2]
        y_plot_fore = g_forefft[:self.n_frame//2]

        self.index_nose = numpy.argmax(y_plot_nose)
        self.index_fore = numpy.argmax(y_plot_fore)

        self.bpm_a = self.xf[self.index_nose] * 60
        self.bpm_b = self.xf[self.index_fore] * 60

        #Gets the pulse
        self.window_hr = self.get_pulse(self.bpm_a)
        print(self.bpm_a,"BPM NARIZ")
        print(self.bpm_b,"BPM FORE")
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

    def bandpass_filter(self, data, lowcut, highcut, fs):

        order = 5.0
        nyq = 0.5 * fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = butter(order,[low,high],btype='band')
        y = lfilter(b,a,data)
        return y

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


    def main(self):

        try:
            self.frames = bridge.imgmsg_to_cv2(self.current_image,"bgr8")
        except CvBridgeError as e:
            print(e)

        try:
            self.depth = bridge.imgmsg_to_cv2(self.current_depth,desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
        self.prev_mouth_status = self.mouth_status

        self.current = time.time()
        self.gray = cv2.cvtColor(self.frames, cv2.COLOR_BGR2GRAY)
        self.gray = cv2.equalizeHist(self.gray) ##Histogram equalization
        rects = self.detector(self.gray, 0)
        size = self.frames.shape

        if not rects:

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (15,20)
            fontScale = 0.65
            fontColor = (255,0,0)
            lineType  = 1
            print("NO SE DETECTA NINGUNA CARAAAAAA", self.contador_detec)
            self.contador_detec += 1
            #cv2.putText(self.frames,'NO DETECTION!', bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            if self.flag_300 ==0:
                self.data_buffernose_rs=[]
                self.data_buffernose_gs=[]
                self.data_buffernose_bs=[]
                self.data_bufferforehead_rs=[]
                self.data_bufferforehead_gs=[]
                self.data_bufferforehead_bs=[]
                self.cont=0
            else:
                pass
            self.time_v = []
            #self.visualizacion(self.frames)
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
            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
            for (x, y) in shape[27:35]:
                cv2.circle(self.frames, (x, y), 1, (0, 255, 0), -1)
            #Get the ROI of the nose
            vector_a=numpy.array([shape[27:35]])
            (x_r,y_r,w_r,h_r)=cv2.boundingRect(vector_a)
            roi_visualizacion_nose = self.frames[y_r:y_r+h_r,x_r:x_r+w_r]
            #roi_visualizacion_nose = imutils.resize(roi_visualizacion_nose, width=100)

            # draw over the (x,y) coordinates of the rigth and left eyebrow
            (x_ci,y_ci) = shape[20]
            cv2.circle(self.frames, (x_ci,y_ci), 1, (0,255,0),-1)
            (x_cd,y_cd) = shape[23]
            cv2.circle(self.frames, (x_cd,y_cd), 1, (0,255,0),-1)

            # draw the ROI of the forehead
            cv2.circle(self.frames,(x_ci,y_ci-10), 1,(255,0,0),-1)
            cv2.circle(self.frames,(x_cd,y_cd-10),1, (255,0,0),-1)
            cv2.rectangle(self.frames,(x_ci,y_ci-25),(x_cd,y_cd-10),(255,0,0),1)
            roi_visualizacion_forehead = self.frames[y_ci-25:y_cd-10, x_ci:x_cd]

            # extract each channel from the nose ROI
            nose_r = roi_visualizacion_nose[:,:,0]
            nose_g = roi_visualizacion_nose[:,:,1]
            nose_b = roi_visualizacion_nose[:,:,2]

            # extract each channel from the forehead ROI
            forehead_r = roi_visualizacion_forehead[:,:,0]
            forehead_g = roi_visualizacion_forehead[:,:,1]
            forehead_b = roi_visualizacion_forehead[:,:,2]

            #Median filter of each channel
            nose_r = cv2.medianBlur(nose_r,3)
            nose_g = cv2.medianBlur(nose_g,3)
            nose_b = cv2.medianBlur(nose_b,3)
            forehead_r = cv2.medianBlur(forehead_r,3)
            forehead_g = cv2.medianBlur(forehead_g,3)
            forehead_b = cv2.medianBlur(forehead_b,3)

            # Spatial average of each channel
            self.m_nr = cv2.mean(nose_r)
            self.m_ng = cv2.mean(nose_g)
            self.m_nb = cv2.mean(nose_b)
            self.m_fr = cv2.mean(forehead_r)
            self.m_fg = cv2.mean(forehead_g)
            self.m_fb = cv2.mean(forehead_b)

            #Extract the left and rigth coordinates from left and rigth eye,then calculate eye ratio
            #And extract the coordinates of the mouth
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = self.eye_ratio(leftEye)
            rightEAR = self.eye_ratio(rightEye)
            mouAR = self.mouth_ratio(mouth)
            eye_r = (leftEAR + rightEAR) / 2.0
            angle_pitch , yaw , roll = self.face_orientation(size,shape)
            self.actual_time = (time.time()-self.current)
            self.time_v.append(self.actual_time)
            #print(self.actual_time)
            self.msg.eye_area = eye_r
            self.msg.mouth_area = mouAR
            self.msg.angle_pitch = angle_pitch
            self.msg.angle_yaw = yaw
            self.msg.angle_roll = roll
            self.msg.heart_rate = self.pulso_pantalla
            self.info.publish(self.msg)
            self.info.publish(self.msg)


            if self.flag_300 == 0:
                self.add_buffer_data()
                if self.cont == 200:
                    self.normalizacion()
                    self.cont = 0
                else:
                    pass
            else:
                self.add_buffer_data_2()

def main():

    rospy.init_node('listener',anonymous=True)
    cam = Camera()
    rospy.loginfo('Node started')

    while not rospy.is_shutdown():
        if enable_img is True:
            #print("Entre")
            cam.main()
    #rospy.sleep(0.01)
    rospy.spin()
#    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
