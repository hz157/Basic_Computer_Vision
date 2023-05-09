#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Project: Face recognition 
# @ File: human.py
# @ Time: 18/2/2023 上午11:15
# @ Author: hz157
# @ Github: https://github.com/hz157

import threading
import time
import cv2
from config import HUMAN_FACE_MODEL, HUMAN_FACE_CLASSIFIER
from human_info import INFORMATION


class HumanFace:
    """
    Human Face features
    """
    def __init__(self, camera):
        self.frame = None
        self.camera = camera
        realTimeScanning = threading.Thread(target=self.lunchCamera)
        realTimeRecognize = threading.Thread(target=self.recognize)
        realTimeScanning.start()
        time.sleep(2)  # delay 2s lunch thread no.2
        realTimeRecognize.start()

    def lunchCamera(self):
        """
        lunch Camera Capture Image
        :return: No Return
        """
        cap = cv2.VideoCapture(self.camera)
        ret, self.frame = cap.read()
        while ret:
            ret, self.frame = cap.read()

    def recognize(self):
        """
        Judgement Camera Capture Image Whether a human is recognised
        :return: Return dict data
        """
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        while True:
            # load data set
            if self.frame is not None:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                recognizer.read(HUMAN_FACE_MODEL)
                # Image
                grey = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                # Classifier
                face_detector = cv2.CascadeClassifier(HUMAN_FACE_CLASSIFIER)
                face = face_detector.detectMultiScale(grey)
                for x, y, w, h in face:
                    label, confidence = recognizer.predict(grey[y:y + h, x:x + w])
                    if confidence > 60:
                        print(confidence)
                        for i in range(len(INFORMATION)):
                            if str(INFORMATION[i]['label']) == str(label):
                                text = INFORMATION[i]['name']
                                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(self.frame, text, (x+w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        print("No data matched")

                cv2.imshow("Face Recognition", self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

