#!/usr/bin/env python
import rospy
import numpy as np
import dlib
import cv2
import jade
import tf
from camara.msg import driver_info
from face_recognition.msg import Faces
from geometry_msgs.msg import Quaternion , PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from imutils import face_utils
import time
import os
import sys
from scipy.spatial import distance as dist
import math
import threading
from scipy.signal import butter, lfilter
from stabilizer import Stabilizer

correr = False
bridge = CvBridge()

class Pulse(object):
    def __init__(self):

        """Subscribers"""
        rospy.Subscriber("/faces_images" , Faces , self.callback_face,queue_size=1)
        rospy.Subscriber("/any_detection" , Bool , self.callback_flag,queue_size=1)

        """Publishers """
        self.quaternion = rospy.Publisher("quaternion", PoseStamped  , queue_size = 1)
        self.q = PoseStamped()
        self.q.header.frame_id = "map"
        self.info = rospy.Publisher("driver_info", driver_info,queue_size = 1)
        self.msg_driver = driver_info()

        """Node Parameters"""
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'shape_predictor_68_face_landmarks.dat'))
        self.model68 = (os.path.join(os.path.dirname(sys.path[0]), 'scripts', 'model.txt'))
        self.size = (rospy.get_param('~img_heigth', 480),rospy.get_param('~img_width', 640))
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
        self.mouth_thresh = 0.55
        self.num_frames = 2
        self.blink_counter = 0
        self.total = 0
        self.eyes_open = 0.0
        self.eyes_closed = 0.0
        self.perclos = 0.0
        self.mouth_status = False
        self.img = None
        self.flag_detec = None

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

        self.main()

    def callback_face(self,datos):

        if self.img is None:

            self.msg = datos.faces_images
            try:
                self.img = bridge.imgmsg_to_cv2(self.msg[0],desired_encoding="passthrough")
            except CvBridgeError as e:
                print(e)

    def callback_flag(self,datos):
        if self.flag_detec is None:
            self.flag_detec = datos.data


    def visualizacion(self,frame):

        cv2.imshow("Frame",frame)
        cv2.waitKey(1)

    def bandpass_filter(self, data, lowcut, highcut, fs):

        order = 5.0
        nyq = 0.5 * fs
        low = lowcut/nyq
        high = highcut/nyq
        b,a = butter(order,[low,high],btype='band')
        y = lfilter(b,a,data)
        return y

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

    def  eye_ratio(self,eye):

        A = dist.euclidean(eye[1],eye[5])
        B = dist.euclidean(eye[2],eye[4])
        C = dist.euclidean(eye[0],eye[3])
        eye_ratio = (A+B)#/(C * 2.0)

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

    def face_orientation(self,size,landmarks):

        image_points = np.array([
                            (landmarks[33][0],landmarks[33][1]),     # Nose
                            (landmarks[8][0],landmarks[8][1]),   # Chin
                            (landmarks[36][0],landmarks[36][1]),     # Left eye corner
                            (landmarks[45][0],landmarks[45][1]),     # Right eye  corner
                            (landmarks[48][0],landmarks[48][1]),     # Left corner Mouth
                            (landmarks[54][0],landmarks[54][1])      # Right corner Mouth
                            ], dtype= "double")

                            #Anthopological values
        model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

        #Values of the camera matrix
        center = (size[1]/2, size[0]/2)
        focal_length = center[0] / np.tan(60/2 * np.pi / 180)
        camera_matrix = np.array(
                     [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype = "double"
                     )


        print(model_points.shape , "SHAPE MODEL")
        print(image_points.shape , "SHAPE MODEL ")

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        axis = np.float32([[500,0,0],
                      [0,500,0],
                      [0,0,500]])

        imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        roll = math.degrees(math.asin(math.sin(roll)))

        return pitch , yaw , roll

    def add_buffer_data(self):

        self.data_buffernose_gray.append(self.nose_gray[0])
        self.data_bufferforehead_gray.append(self.forehead_gray[0])
        self.cont += 1

    def add_buffer_data_2(self):

        self.data_buffernose_gray[self.cont]=(self.nose_gray[0])
        self.data_bufferforehead_gray[self.cont]=(self.forehead_gray[0])
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
        #print(np.shape(self.time_v))
        self.flag_300 =1
        #self.n_frame=len(self.data_buffernose_gs)
        self.n_frame = len(self.data_buffernose_gray)
        self.time = np.mean(self.time_v)
        self.fs = 1.0/self.time
        self.xf = np.linspace(0.0,(self.fs/2),self.n_frame/2)

        # Normalizacion datos
        n_datanose_gray = (self.data_buffernose_gray - np.mean(self.data_buffernose_gray))/np.std(self.data_buffernose_gray)
        n_datafore_gray = (self.data_bufferforehead_gray - np.mean(self.data_bufferforehead_gray))/np.std(self.data_bufferforehead_gray)

        # Creating matrix for ICA
        ica_nose = n_datanose_gray
        ica_fore = n_datafore_gray

        self.ica_both = np.zeros((2,len(n_datanose_gray)))
        self.ica_both[0,:] = ica_nose
        self.ica_both[1,:] = ica_fore
        thread_a  = threading.Thread(target = self.ICA)
        thread_a.start()

    def ICA(self):

        # Applying ICA
        nose_ica = jade.main(self.ica_both)

        # Transpose of ICA result
        nose_ica=nose_ica.T

        #nose_green = nose_ica[1,:]
        nose_green = nose_ica[0,:]
        nose_green = np.ravel(nose_green)
        nose_green = np.hamming(self.n_frame) * nose_green

        #Bandpass filter on the green channel on nose and forehead
        nose_green= self.bandpass_filter(nose_green,0.95,2.0,self.fs)
        fore_ica = self.bandpass_filter(nose_ica[1,:],0.95,2.0,self.fs)

        #Fourier transform
        g_nosefft= np.absolute(np.fft.fft(nose_green, norm="ortho"))
        f_forefft= np.absolute(np.fft.fft(fore_ica, norm="ortho"))
        g_nosefft= np.ravel(g_nosefft)
        g_forefft= np.ravel(f_forefft)

        #Only takes de half of the data
        y_plot_nose = g_nosefft[:self.n_frame//2]
        y_plot_fore = g_forefft[:self.n_frame//2]

        self.index_nose = np.argmax(y_plot_nose)
        self.index_fore = np.argmax(y_plot_fore)

        self.bpm_a = self.xf[self.index_nose] * 60
        self.bpm_b = self.xf[self.index_fore] * 60

        #Gets the pulse
        self.window_hr = self.get_pulse(self.bpm_a)
        print(self.bpm_a,"BPM componente 1")
        print(self.bpm_b,"BPM componente 2")

    def get_pulse(self,ad_hr):

        # T + 1 pulse adquiction
        self.pulso_guardado = self.pulso_adquirido
        # T pulse adquiction
        self.pulso_adquirido = ad_hr
        comparation = np.absolute(self.pulso_guardado - self.pulso_adquirido)
        if comparation < 1:
            #Displayed pulse
            self.pulso_pantalla = (self.pulso_guardado + self.pulso_adquirido)/2

        return self.pulso_pantalla

    def main(self):
        self.current = time.time()
        while not rospy.is_shutdown():


            if not(self.img is None) and (not(self.flag_detec) is None):
                self.actual_time = time.time() - self.current
                self.current = time.time()
                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                self.prev_mouth_status = self.mouth_status

                if self.flag_detec is False:

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (15,20)
                    fontScale = 0.4
                    fontColor = (255,0,0)
                    lineType  = 1
                    cv2.putText(self.img,'NO DETECTION!', bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
                    if self.flag_300 ==0:
                        self.data_buffernose_gray = []
                        self.data_bufferforehead_gray = []
                        self.cont=0

                    else:
                        pass
                    self.time_v = []
                    self.visualizacion(self.img)
                    #self.visualizacion_ros(self.frames)
                    self.flag = 1

                else:

                    self.gray = cv2.equalizeHist(self.img)
                    recta = dlib.rectangle(0,0,self.gray.shape[1],self.gray.shape[0 ])
                    shape = self.predictor(self.gray,recta)
                    shape = face_utils.shape_to_np(shape)

                    for (x, y) in shape[27:35]:
                        cv2.circle(self.img, (x, y), 1, (0, 255, 0), -1)

                    vector_a=np.array([shape[27:35]])
                    (x_r,y_r,w_r,h_r)=cv2.boundingRect(vector_a)
                    roi_visualizacion_nose = self.img[y_r:y_r+h_r,x_r:x_r+w_r]

                    for (x, y) in shape[48:60]:
                        cv2.circle(self.img, (x, y), 1, (0, 255, 0), -1)

                    (x_ci,y_ci) = shape[20]
                    cv2.circle(self.img, (x_ci,y_ci), 1, (0,255,0),-1)
                    (x_cd,y_cd) = shape[23]
                    cv2.circle(self.img, (x_cd,y_cd), 1, (0,255,0),-1)

                    cv2.circle(self.img,(x_ci,y_ci-10), 1,(255,0,0),-1)
                    cv2.circle(self.img,(x_cd,y_cd-10),1, (255,0,0),-1)
                    cv2.rectangle(self.img,(x_ci,y_ci-25),(x_cd,y_cd-10),(255,0,0),1)
                    roi_visualizacion_forehead = self.img[y_ci-25:y_cd-10, x_ci:x_cd]

                    nose_gray = roi_visualizacion_nose
                    forehead_gray = roi_visualizacion_forehead

                    nose_gray = cv2.medianBlur(nose_gray,3)
                    forehead_gray = cv2.medianBlur(forehead_gray,3)

                    self.nose_gray = cv2.mean(nose_gray)
                    self.forehead_gray = cv2.mean(forehead_gray)

                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    mouth = shape[mStart:mEnd]
                    leftEAR = self.eye_ratio(leftEye)
                    rightEAR = self.eye_ratio(rightEye)
                    mouAR = self.mouth_ratio(mouth)
                    eye_r = (leftEAR + rightEAR) / 2.0

                    if eye_r < 8.3:
                        self.blink_counter += 1
                        self.eyes_closed += 1

                        if self.blink_counter > 50:
                            print("SE ESTA QUEDANDO DORMIDOOOOOOOOOOOOOOOOOOOOOOOOO")

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
                            print("SE ESTA QUEDANDO DORMIDOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                        self.eyes_open = 0.0
                        self.eyes_closed = 0.0

                    if mouAR > self.mouth_thresh:
                        self.mouth_status = True

                    else:
                        self.mouth_status = False

                    if self.prev_mouth_status == True and self.mouth_status == False:
                        self.cont_yawn +=1

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

                    #angle_pitch , yaw , roll = self.face_orientation(self.size,shape)
                    #self.q.pose.orientation.x , self.q.pose.orientation.y , self.q.pose.orientation.z , self.q.pose.orientation.w = tf.transformations.quaternion_from_euler(yaw, angle_pitch, roll)
                    #self.quaternion.publish(self.q)
                    self.visualizacion(self.img)
                    self.time_v.append(self.actual_time)
                    #print(self.actual_time)

                    """visualizacion de datos"""

                    print("PITCH:" , euler_angle[0] ,"YAW:", euler_angle[1] , "ROLL:" , euler_angle[2])
                    print("HEART RATE:" , self.pulso_pantalla)
                    print("NUM PARPADEOS",self.total , "NUM BOSTEZOS" , self.cont_yawn  )

                    try:
                        self.img_pub = bridge. cv2_to_imgmsg(self.img,encoding="passthrough")
                    except CvBridgeError as e:
                        print(e)

                    self.msg_driver.imagen_bag = self.img_pub
                    self.msg_driver.eye_area = eye_r
                    self.msg_driver.mouth_area = mouAR
                    self.msg_driver.angle_pitch = euler_angle[0]
                    self.msg_driver.angle_yaw = euler_angle[1]
                    self.msg_driver.angle_roll = euler_angle[2]
                    self.msg_driver.heart_rate = self.pulso_pantalla
                    self.msg_driver.perclos = self.perclos
                    self.info.publish(self.msg_driver)

                    if self.flag_300 == 0:
                        self.add_buffer_data()
                        if self.cont == 200:
                            self.normalizacion()
                            self.cont = 0
                        else:
                            pass
                    else:
                        self.add_buffer_data_2()

                self.img = None
                self.flag_detec = None


if __name__ == '__main__':

    try:
        rospy.init_node("adas_node", anonymous = True)
        pulse = Pulse()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
