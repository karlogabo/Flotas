import numpy as np
import dlib
import cv2
import time

class FeatureExtraction(object):
    def __init__(self):
        self.video_service(0)

    def video_service(self, cam_n):
        PREDICTOR_PATH = "/home/frs-zuca/frs_ws/src/face_recognition/include/shape_predictor_68_face_landmarks.dat"

        FACE_RECOGNITION_MODEL_PATH = "/home/frs-zuca/frs_ws/src/face_recognition/include/dlib_face_recognition_resnet_model_v1.dat"
        faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

        try:
            winName = "Fast Facial Landmark Detector"
            cap = cv2.VideoCapture(cam_n)
            ret, im = cap.read()
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(PREDICTOR_PATH)
            while (True):
                ret, im = cap.read()
                faces = detector(im, 0)
                faceDescriptors = None
                for i in xrange(len(faces)):
                    t = time.time()
                    shape = predictor(im, faces[i])
                    print(time.time()-t)
                    t = time.time()
                    faceDescriptor = faceRecognizer.compute_face_descriptor(im, shape)
                    print(time.time()-t)

                cv2.imshow('frame',im)

        except:
            raise
if __name__== "__main__":
    try:
        feature_extraction = FeatureExtraction()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


    # def __init__(self):
    #     self.face_detector = dlib.get_frontal_face_detector()
    #     self.face_predictor = dlib.shape_predictor("/home/frs-zuca/frs_ws/src/face_recognition/include/shape_predictor_68_face_landmarks.dat")
    #     self.face_descriptor = dlib.face_recognition_model_v1("/home/frs-zuca/frs_ws/src/face_recognition/include/dlib_face_recognition_resnet_model_v1.dat")
    #
    #     self.main()
    #
    # def main(self):
    #     cap = cv2.VideoCapture(0)
    #     SKIP_FRAMES = 2
    #     count = 0
    #     faceRecognizer = dlib.face_recognition_model_v1("/home/frs-zuca/frs_ws/src/face_recognition/include/dlib_face_recognition_resnet_model_v1.dat")
    #     while True:
    #         ret, frame = cap.read()
    #         if (count % SKIP_FRAMES == 0):
    #             count = 0
    #             faces = self.face_detector(frame, 0)
    #             for face in faces:
    #
    #                 t = time()
    #                 shape = self.face_predictor(frame,face)
    #                 print(time()-t)
    #
    #                 t = time()
    #                 descriptor = faceRecognizer.compute_face_descriptor(frame, shape)
    #                 print(time()-t)
    #                 # print(type(face_descriptor),type(face_descriptor[1]))
    #                 shape = np.array([(int(shape.part(i).x),int(shape.part(i).y)) for i in range(68)])
    #                 for point in shape:
    #                     cv2.circle(frame,tuple(point),2,(0,0,255),-1)
    #
    #
    #         count += 1
    #
