import time

import cv2
import dlib
import imutils
import numpy as np
import requests
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist


class DrowsinessDetector:
    def __init__(self, driver_info, shape_predictor, url, webcam):
        self.EYE_ASPECT_RATIO_THR = 0.2
        self.EYE_ASPECT_RATIO_CNSCTV_FRAMES_1 = 32
        self.EYE_ASPECT_RATIO_CNSCTV_FRAMES_2 = 64
        
        self.driver_info = driver_info
        self.url = url
        self.webcam = webcam
        self.alarm_1 = False
        self.alarm_2 = False

        self.counter = 0

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor)

        (self.l_start, self.l_end) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.r_start, self.r_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

    def __eye_aspect_ratio(self, eye):
        eye_A = dist.euclidean(eye[1], eye[5])
        eye_B = dist.euclidean(eye[2], eye[4])
        eye_C = dist.euclidean(eye[0], eye[3])

        aspect_ratio = (eye_A + eye_B) / (2.0 * eye_C)
        return aspect_ratio

    def execute(self):
        vs = VideoStream(src = self.webcam).start()
        time.sleep(1.0)

        while (True):
            frame = vs.read()
            frame = imutils.resize(frame, width = 450)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray_frame, 0)

            for rect in rects:
                shape = self.predictor(gray_frame, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[self.l_start : self.l_end]
                right_eye = shape[self.r_start : self.r_end]
                left_eye_ar = self.__eye_aspect_ratio(left_eye)
                right_eye_ar = self.__eye_aspect_ratio(right_eye)

                eye_ar = (left_eye_ar + right_eye_ar) / 2.0

                left_eye_hull= cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

                if (eye_ar < self.EYE_ASPECT_RATIO_THR):
                    self.counter += 1

                    if(self.counter >= self.EYE_ASPECT_RATIO_CNSCTV_FRAMES_1):
                        alert_level = 1
                        self.driver_info['alert_level'] = alert_level
                        if not self.alarm_1:
                            self.driver_info['curr_time'] = int(time.time())
                            r = requests.post(url = self.url, json = self.driver_info)
                            print(r.content)
                            self.alarm_1 = True
                        
                        if (self.counter >= self.EYE_ASPECT_RATIO_CNSCTV_FRAMES_2):
                            alert_level = 2
                            self.driver_info['alert_level'] = alert_level
                            if not self.alarm_2:
                                self.driver_info['curr_time'] = int(time.time())
                                r = requests.post(url = self.url, json = self.driver_info)
                                print(r.content)
                                self.alarm_2 = True

                        cv2.putText(frame, "DRIVER DROWSINESS ALERT: LEVEL {}".format(alert_level), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    self.counter = 0
                    self.alarm_1 = False
                    self.alarm_2 = False


            cv2.imshow('Frame', frame)
            key = cv2.waitKey(1) & 0xFF

            if (key == ord('q')):
                break

        cv2.destroyAllWindows()
        vs.stop()
