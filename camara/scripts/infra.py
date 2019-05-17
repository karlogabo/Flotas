#!/usr/bin/env python
import rospy
import numpy as np
import dlib
import cv2
import jade
# import tf
#import matplotlib.pyplot as plt
from camara.msg import driver_info
#from face_recognition.msg import Faces
from geometry_msgs.msg import Quaternion , PoseStamped
from std_msgs.msg import Bool
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from imutils import face_utils
import time
import os
import sys
from scipy.spatial import distance as dist
from scipy.signal import savgol_filter as sg
from scipy.signal import firwin as fw
from scipy.signal import freqz as fq
import math
import threading
#from spectrum import tools as stools
#from spectrum import pburg
from scipy.signal import butter, lfilter
from stabilizer import Stabilizer

current_image = Image()
current_depth = Image()
bridge = CvBridge()
enable_img = False
enable_infra = False
enable_flag = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/home/innovacion/ADAS_workspace/src/camara/scripts/adas.avi', fourcc, 20.0, (640, 480))
# plt.ion()


class Camera(object):
    def __init__(self):

        """ Subscribers """
        rospy.Subscriber("/camera/depth/image_rect_raw" , Image , self.callback_depth, queue_size=1)
        rospy.Subscriber("/camera/infra1/image_rect_raw", Image,  self.callback_infra, queue_size=1)
        rospy.Subscriber("/flag_pub", Float32, self.callback_flag, queue_size = 1)

        """ Publishers """
        self.img_bag_pub = rospy.Publisher("Image_bag" , Image , queue_size = 1 )
        self.info = rospy.Publisher("driver_info", driver_info, queue_size = 1)
        self.msg_driver = driver_info()
        self.quaternion = rospy.Publisher("quaternion", PoseStamped  , queue_size = 1)
        self.q = PoseStamped()
        self.q.header.frame_id = "map"
        self.heart_rate_b = rospy.Publisher("heart_rate_b" , Float32, queue_size = 1)
        self.heart_rate_b_msg = Float32()

        """ Node Parameters """
        self.cont_video = 0
        self.data_buffernose_gray = [0]*200
        self.data_bufferforehead_gray = [0]*200
        self.matrix_v = [0]
        self.nose_ica_g=[0]
        self.fore_ica_g=[0]
        self.time_v=[]
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'shape_predictor_68_face_landmarks.dat'))
        self.model68 = (os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'model.txt'))
        self.bpm_a = 0
        self.bpm_a_b = 0
        self.window_hr=0
        self.yawn_state = False
        self.window_hr_b=0
        self.pulso_pantalla = 0
        self.pulso_guardado = 0
        self.pulso_adquirido = 0
        self.pulso_pantalla_b = 0
        self.pulso_guardado_b = 0
        self.pulso_adquirido_b = 0
        self.pitch_counter = 0
        self.cont_state = 0
        self.cont = 0
        self.cont2 = 0
        self.yaw_counter = 0
        self.cont_yawn = 0
        self.yawn_counter = 0
        self.flag = 0
        self.flag_300 = 0
        self.eye_thresh = 0.3
        self.mouth_thresh = 0.60
        self.num_frames = 2
        self.blink_counter = 0
        self.total = 0
        self.eyes_open = 0.0
        self.eyes_closed = 0.0
        self.perclos = 0.0
        self.mouth_status = False

        #self.current_depth = []
        #cv2.namedWindow('frame')
        #self.img_publisher = rospy.Publisher("topico_imagen", Image)

        """Parameters for head pose estimation"""

        self.model_points_68 = self._get_full_model_points()
        self.size=(480, 640)
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")
        self.dist_coeefs = np.zeros((4, 1))
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

        self.pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

        """Graph Parameters"""
        # self.data_buffernose_gray = [0]*200
        # self.data_bufferforehead_gray = [0]*200
        # self.blinks = [0]*40
        # self.umbral = [30]*40
        # self.blinks_sg = [0]*40
        # self.d_blinks = [0]*39
        # self.mirar = [0]*1
        # self.fig = plt.figure()
        # self.ax1 = self.fig.add_subplot(211)
        # self.ax1.set_ylim(10 , 55)
        # self.line,  = self.ax1.plot(range(40),self.blinks, 'r-')
        # plt.ylabel("Normalized Distance (pixels)")
        # plt.xlabel("Number of frames")
        # plt.title("Eyeblink detection")
        # self.ax1.plot(range(40),self.umbral, 'b-')

        # self.ax2 = self.fig.add_subplot(212)
        # self.ax2.set_ylim(5 , 20)
        # self.line2, = self.ax2.plot(range(40),self.blinks_sg,'b-')
        # thismanager = plt.get_current_fig_manager()
        # thismanager.window.wm_geometry("+700+1")

    def callback_rgb(self,datos):

        #Callback for acquiring rgb data from the realsense
        #global enable_img
        #enable_img = True
        self.current_image = datos
        # print('callback')

    def sma(self,data,window):

        sma = np.convolve(data, np.ones(window), "valid")/window
        data = data[int(window-1):]
        sma_window = data - sma

        return sma_window

    def _get_full_model_points(self, filename='model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(self.model68) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:, -1] *= -1

        return model_points

    def solve_pose_by_68_points(self, image_points):

        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """
        image_points = np.float32(image_points)
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

        global enable_img
        enable_img = True
        self.current_depth = datos

    def toQuaternion(self,yaw,pitch,roll):

        cy = math.cos(yaw * 0.5);
        sy = math.sin(yaw * 0.5);
        cp = math.cos(pitch * 0.5);
        sp = math.sin(pitch * 0.5);
        cr = math.cos(roll * 0.5);
        sr = math.sin(roll * 0.5);
        self.q.pose.orientation.w = cy * cp * cr + sy * sp * sr;
        self.q.pose.orientation.x = cy * cp * sr - sy * sp * cr;
        self.q.pose.orientation.y= sy * cp * sr + cy * sp * cr;
        self.q.pose.orientation.z= sy * cp * cr - cy * sp * sr;

    def callback_infra(self, datos):

        global enable_infra
        enable_infra = True
        self.current_infra = datos

    def callback_flag(self, datos):

        global enable_flag
        if datos.data == 1.0:
            enable_flag = True
        else:
            enable_flag = False


    def detect_near_face(self, rects):

        dist = []
        for rect in rects:
            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)
            vector_b = np.array([shape[27:35]])
            (x_rr,y_rr,w_rr,h_rr)=cv2.boundingRect(vector_b)
            roi_depth = self.depth[y_rr : y_rr + h_rr , x_rr : x_rr + w_rr]
            #roi_depth = self.depth[rect.top():rect.bottom(), rect.left():rect.right()]
            mean_distance = cv2.mean(roi_depth)
            dist.append(mean_distance[0])

        index = np.argmin(dist)
        rectss = rects.pop(index)

        return rectss

    def distance(self, shape):

        vector_b = np.array([shape[27:35]])
        (x_rr,y_rr,w_rr,h_rr)=cv2.boundingRect(vector_b)
        roi_depth = self.depth[y_rr : y_rr + h_rr , x_rr : x_rr + w_rr]
        mean_distance = cv2.mean(roi_depth)

        if mean_distance[0] > 3500 and mean_distance[0] < 4000:
            val = 35
        elif mean_distance[0] > 2500 and mean_distance[0] < 3500:
            val = 45
        elif mean_distance[0] > 2000 and mean_distance[0] < 2500:
            val = 55
        else:
            val = 25

        return val, mean_distance

    def close_camera(self):

        self.video_capture.release()

    def visualizacion(self, frame):

        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

    def  eye_ratio(self,eye):

        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        eye_ratio = (A+B)#/(C * 2.0)
        # print(eye_ratio)

        return eye_ratio

    def mouth_ratio(self,mou):

        #Horizontal
        a  = dist.euclidean(mou[0], mou[6])
        #Vertical
        b  = dist.euclidean(mou[3], mou[9])
        c  = dist.euclidean(mou[2], mou[10])
        d  = dist.euclidean(mou[4], mou[8])
        e  = (b+c+d)/3

        mouth_ratio = e/a

        return mouth_ratio

    def get_pulse(self, ad_hr):

        # T + 1 pulse adquiction
        self.pulso_guardado = self.pulso_adquirido
        # T pulse adquiction
        self.pulso_adquirido = ad_hr
        comparation = np.absolute(self.pulso_guardado - self.pulso_adquirido)
        if comparation < 1:
            #Displayed pulse
            self.pulso_pantalla = (self.pulso_guardado + self.pulso_adquirido)/2

        return self.pulso_pantalla

    def get_pulse_b(self, ad_hr):

        # T + 1 pulse adquiction
        self.pulso_guardado_b = self.pulso_adquirido_b
        # T pulse adquiction
        self.pulso_adquirido_b = ad_hr
        comparation = np.absolute(self.pulso_guardado_b - self.pulso_adquirido_b)
        if comparation < 1:
            #Displayed pulse
            self.pulso_pantalla_b = (self.pulso_guardado_b + self.pulso_adquirido_b)/2

        return self.pulso_pantalla_b

    def add_buffer_data(self):

        self.data_buffernose_gray.pop(0)
        self.data_bufferforehead_gray.pop(0)
        self.data_buffernose_gray.append(self.nose_gray[0])
        self.data_bufferforehead_gray.append(self.forehead_gray[0])
        self.cont += 1

    def add_buffer_data_2(self):

        self.data_buffernose_gray.pop(0)
        self.data_buffernose_gray.append(self.nose_gray[0])
        self.data_bufferforehead_gray.pop(0)
        self.data_bufferforehead_gray.append(self.forehead_gray[0])
        self.cont += 1
        self.cont2 += 1

        if self.cont > 199:
            self.cont = 0
        else:
            pass
        if self.cont2 == 20: #En vez de  5: ESTA EN 10
            self.cont2 = 0
            self.normalizacion()
            print("")
        else:
            pass

    def bandpass_filter(self, data, lowcut, highcut, fs):

        order = 5.0 #5.0 el bag PULSE_A
        nyq = 0.5 * fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = butter(order, [low,high], btype='band')
        y = lfilter(b, a, data)

        return y

    def normalizacion(self):

        self.flag_300 =1
        self.n_frame = len(self.data_buffernose_gray)
        self.time = np.mean(self.time_v)
        self.fs = 1.0/self.time
        self.data_buffernose_gray_b = self.sma(self.data_buffernose_gray, 5)
        self.data_bufferforehead_gray_b = self.sma(self.data_bufferforehead_gray, 5)
        self.xf = np.linspace(0.0,(self.fs/2),self.n_frame/2)

        # Normalizacion datos
        n_datanose_gray = (self.data_buffernose_gray - np.mean(self.data_buffernose_gray))/np.std(self.data_buffernose_gray)
        n_datafore_gray = (self.data_bufferforehead_gray - np.mean(self.data_bufferforehead_gray))/np.std(self.data_bufferforehead_gray)
        n_datanose_gray_b = (self.data_buffernose_gray_b)/np.std(self.data_buffernose_gray_b)
        n_datafore_gray_b = (self.data_bufferforehead_gray_b)/np.std(self.data_bufferforehead_gray_b)

        # Creating matrix for ICA
        ica_nose = n_datanose_gray
        ica_fore = n_datafore_gray
        ica_nose_b = n_datanose_gray_b
        ica_fore_b = n_datafore_gray_b
        self.ica_both = np.zeros((2,len(n_datanose_gray)))
        self.ica_both_b = np.zeros((2,len(n_datanose_gray_b)))
        self.ica_both[0,:] = ica_nose
        self.ica_both[1,:] = ica_fore
        self.ica_both_b[0,:] = ica_nose_b
        self.ica_both_b[1,:] = ica_fore_b
        thread_a  = threading.Thread(target = self.ICA)
        thread_a.start()

    def ICA(self):

        # Applying ICA
        nose_ica = jade.main(self.ica_both)
        nose_ica_b = self.ica_both_b
        nose_ica=nose_ica.T

        nose_green = nose_ica[0,:]
        nose_green = np.ravel(nose_green)
        nose_green = np.hamming(len(nose_green)) * nose_green
        nose_green_b = nose_ica_b[0,:]
        nose_green_b = np.ravel(nose_green_b)
        nose_green_b = np.hamming(len(nose_green_b)) * nose_green_b
        #Bandpass filter on the green channel on nose and forehead
        nose_green= self.bandpass_filter(nose_green,0.90, 1.7,self.fs)  #ESTA E N 0.95 a 2.0
        fore_ica = self.bandpass_filter(nose_ica[1,:],0.90, 1.7,self.fs)
        nose_green_b = self.bandpass_filter(nose_green_b , 0.90 , 1.7, self.fs)
        fore_green_b = self.bandpass_filter(nose_ica_b[1,:], 0.90, 1.7, self.fs)

        #Fourier transform
        g_nosefft= np.absolute(np.fft.fft(nose_green, norm="ortho"))
        f_forefft= np.absolute(np.fft.fft(fore_ica, norm="ortho"))
        g_nosefft_b = np.absolute(np.fft.fft(nose_green_b, norm="ortho"))
        g_forefft_b = np.absolute(np.fft.fft(fore_green_b, norm="ortho"))
        g_nosefft= np.ravel(g_nosefft)
        g_forefft= np.ravel(f_forefft)
        g_nosefft_b = np.ravel(g_nosefft_b)
        g_forefft_b = np.ravel(g_forefft_b)

        #Only takes de half of the data
        y_plot_nose = g_nosefft[:self.n_frame//2]
        y_plot_fore = g_forefft[:self.n_frame//2]
        y_plot_nose_b = g_nosefft_b[:self.n_frame//2]
        y_plot_fore_b = g_forefft_b[:self.n_frame//2]
        self.index_nose = np.argmax(y_plot_nose)
        self.index_fore = np.argmax(y_plot_fore)
        self.index_nose_b = np.argmax(y_plot_nose_b)
        self.index_fore_b = np.argmax(y_plot_fore_b)

        #Calculates the hearth rate
        self.bpm_a = self.xf[self.index_nose] * 60
        self.bpm_b = self.xf[self.index_fore] * 60
        self.bpm_a_b = self.xf[self.index_nose_b] * 60
        self.bpm_b_b = self.xf[self.index_fore_b] * 60

        # cwtmatr , freqs = pywt.cwt(y_plot_nose, len(y_plot_nose)  , "morl", self.fs)
        #Gets the pulse
        self.window_hr = self.get_pulse(self.bpm_a)
        self.window_hr_b = self.get_pulse_b(self.bpm_a_b)
        print(self.bpm_a,"BPM componente 1")
        print(self.bpm_b,"BPM componente 2")
        print(self.bpm_a_b,"BPM componente 1 metodo 2")
        print(self.bpm_b_b,"BPM componente 2 metodo 2")

        #Plot fourier
        # plt.clf()
        # g_burg.plot()
        # plt.plot(xf_burg,(10 * stools.log10(g_burg.psd)),'r')
        # plt.plot(nose_green,'r')
        # #plt.plot(self.xf[(self.xf > 0.70)],y_plot_nose[(self.xf > 0.70)],'r',label="Nose")
        # #plt.plot(self.xf[(self.xf > 0.70)],y_plot_fore[(self.xf > 0.70)],'b',label="Forehead")
        # #plt.title("Red-Nose, Blue-Forehead")
        # plt.show()

    def plotting(self, x1):

        self.blinks.pop(0)
        self.blinks.append(x1)
        # self.blinks_sg = sg(self.blinks, 5, 2)
        # peaks , _ = find_peaks(self.blinks)
        self.line.set_ydata(self.blinks)
        # self.line2.set_ydata(self.blinks_sg)

    def main(self):

        try:
            self.depth = bridge.imgmsg_to_cv2(self.current_depth, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        try:
            self.rojo = bridge.imgmsg_to_cv2(self.current_infra, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        # self.frames = frame
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

        self.prev_mouth_status = self.mouth_status   #MIRRRRRRRRRRRRRRRAAAAAAAAAAAAAAAAA
        self.current = time.time()
        self.gray = self.rojo
        self.gray = cv2.equalizeHist(self.gray) ##Histogram equalization
        rects = self.detector(self.gray, 0)
        size = self.rojo.shape ## REVIEW:

        if not rects:

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (15,20)
            fontScale = 0.65
            fontColor = (255,0,0)
            lineType  = 1
            cv2.putText(self.rojo,'NO DETECTION!', bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
            if not self.flag_300:
                self.data_buffernose_gray = [0]*200
                self.data_bufferforehead_gray = [0]*200
                self.cont=0
            else:
                pass
            self.time_v = []
            self.visualizacion(self.rojo)
            self.flag = 1

        else:

            if len(rects) > 1:
                rect = self.detect_near_face(rects)
            elif len(rects) == 1:
                rect = rects.pop()
            else:
                return

            self.flag = 0
            shape = self.predictor(self.gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
            for (x, y) in shape[27:35]:
                cv2.circle(self.rojo, (x, y), 1, (0, 255, 0), -1)
            #Get the ROI of the nose
            vector_a=np.array([shape[27:35]])
            (x_r,y_r,w_r,h_r)=cv2.boundingRect(vector_a)
            roi_visualizacion_nose = self.rojo[y_r:y_r+h_r,x_r:x_r+w_r]

            for (x, y) in shape[48:60]:
                cv2.circle(self.rojo, (x, y), 1, (0, 255, 0), -1)

            # draw over the (x,y) coordinates of the rigth and left eyebrow
            (x_ci,y_ci) = shape[20]
            cv2.circle(self.rojo, (x_ci, y_ci), 1, (0,255,0), -1)
            (x_cd,y_cd) = shape[23]
            cv2.circle(self.rojo, (x_cd, y_cd), 1, (0,255,0), -1)

            y , _ = self.distance(shape)

            # draw the ROI of the forehead
            cv2.circle(self.rojo,(x_ci, y_ci-10), 1, (255,0,0), -1)
            cv2.circle(self.rojo,(x_cd, y_cd-10), 1, (255,0,0), -1)
            cv2.rectangle(self.rojo, (x_ci, y_ci-y), (x_cd, y_cd-10), (255,0,0), 1)
            roi_visualizacion_forehead = self.rojo[y_ci-y:y_cd-10, x_ci:x_cd]

            nose_gray = roi_visualizacion_nose
            forehead_gray = roi_visualizacion_forehead

            #Median filter of each channel
            # nose_gray = cv2.medianBlur(nose_gray, 3)
            # forehead_gray = cv2.medianBlur(forehead_gray, 3)
            # forehead_gray = cv2.medianBlur(forehead_gray, 3)

            # Spatial average of each channel
            self.nose_gray = cv2.mean(nose_gray)
            self.forehead_gray = cv2.mean(forehead_gray)

            #Extract the left and rigth coordinates from left and rigth eye,then calculate eye ratio
            #And extract the coordinates of the mouth
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[48:61]
            leftEAR = self.eye_ratio(leftEye)
            rightEAR = self.eye_ratio(rightEye)
            mouAR = self.mouth_ratio(mouth)

            # Compute each eye ratio
            eye_r = (leftEAR + rightEAR) / 2.0
            _, distance = self.distance(shape)
            eye_r = ((1/eye_r) / distance[0])*1000000
            # self.plotting(eye_r)
            if eye_r > 28:  #Estaba en 30 el primer back
                self.blink_counter += 1
                self.eyes_closed += 1

                if self.blink_counter > 50:
                    print("SE ESTA QUEDANDO DORMIDO")
                    cv2.putText(self.rojo, 'Alerta por fatiga', (420,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),1)

            else:
                self.eyes_open += 1
                if self.blink_counter >= self.num_frames:
                    self.total += 1
                self.blink_counter = 0

            if (self.eyes_open + self.eyes_closed) > 545:
                self.perclos = (self.eyes_closed/(self.eyes_open+self.eyes_closed))*100
                print(self.eyes_closed,"Num. Frames ojos cerrados")
                print(self.eyes_open,"Num. Frames ojos abiertos")
                print(self.perclos,"PERCLOS")
                if self.perclos > 40.0:
                    print("ALERTA MICROSUENO")
                self.eyes_open = 0.0
                self.eyes_closed = 0.0

            # print (mouAR)
            if mouAR > self.mouth_thresh:
                self.yawn_counter +=1

            else:
                if self.yawn_counter >= 13:
                    self.cont_yawn +=1
                    self.yawn_state = True

                self.yawn_counter = 0

            if self.yawn_state is True:

                cv2.putText(self.rojo, 'Alerta por fatiga', (420,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255),1)
                self.cont_state += 1

                if self.cont_state > 30:

                    self.yawn_state = False
                    self.cont_state = 0


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
            # self.q.pose.orientation.x , self.q.pose.orientation.y , self.q.pose.orientation.z, self.q.pose.orientation.w = tf.transformations.quaternion_from_euler(euler_angle[1], euler_angle[0], 0)
            self.toQuaternion(euler_angle[1],euler_angle[0],euler_angle[2]+180)
            self.quaternion.publish(self.q)

            if euler_angle[0] < -15:
                self.pitch_counter += 1
                if self.pitch_counter > 50:
                    cv2.putText(self.rojo, 'Alerta por desatencion', (380,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255), 1)
            else:
                self.pitch_counter = 0

            if euler_angle[1] < -25 or euler_angle[1] > 25:
                self.yaw_counter += 1
                if self.yaw_counter > 50:
                    cv2.putText(self.rojo, 'Alerta por desatencion', (380,20), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255), 1)
            else:
                self.yaw_counter = 0



            #Put the information into the frame
            cv2.putText(self.rojo, 'Num. Parpadeos:{}' .format(self.total), (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1)
            cv2.putText(self.rojo, 'Num. Bostezos:{}' .format(self.cont_yawn), (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 1)
            cv2.putText(self.rojo, 'HR:{}' .format(self.window_hr), (15,460), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(self.rojo, 'HR_B:{}' .format(self.window_hr_b), (15,430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            cv2.putText(self.rojo, 'PITCH:{}' .format(euler_angle[0]), (490,420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(self.rojo, 'YAW:{}' .format(euler_angle[1]), (490,440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(self.rojo, 'ROLL:{}' .format(euler_angle[2]), (490,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            self.video = cv2.cvtColor(self.rojo,cv2.COLOR_GRAY2BGR)
            out.write(self.video)
            #
            # self.cont_video += 1
            #
            #
            # if self.cont_video > 200:
            #     out.release()
            #     print("Ya acabe el video papa")
            #     self.cont_video = 0


            try:
                self.img_pub = bridge. cv2_to_imgmsg(self.rojo, encoding="passthrough")
            except CvBridgeError as e:
                print(e)

            self.msg_driver.eye_area = eye_r
            self.msg_driver.num_blinks = self.total
            self.msg_driver.mouth_area = mouAR
            self.msg_driver.num_yawn = self.cont_yawn
            self.msg_driver.angle_pitch = euler_angle[0]
            self.msg_driver.angle_yaw = euler_angle[1]
            self.msg_driver.angle_roll = euler_angle[2]
            self.msg_driver.heart_rate = self.window_hr
            # self.msg_driver.heart_rate_b = self.window_hr_b
            self.info.publish(self.msg_driver)
            self.img_bag_pub.publish(self.img_pub)
            # self.heart_rate_b_msg.data = self.window_hr_b
            self.heart_rate_b.publish(self.heart_rate_b_msg.data)
            #Append the data on the buffer and display the video
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()
            self.visualizacion(self.rojo)
            self.msg_driver.perclos = self.perclos
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

def main():

    rospy.init_node('Infrarojo',anonymous=True)
    cam = Camera()
    rospy.loginfo('Node started')

    while not rospy.is_shutdown():

        if enable_infra is True and enable_img is True:
            cam.main()
        else:
            print("Grabacion detenida")

    rospy.spin()

if __name__ == '__main__':

    main()
