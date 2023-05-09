#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Project: Face recognition 
# @ File: training.py
# @ Time: 18/2/2023 下午2:18
# @ Author: hz157
# @ Github: https://github.com/hz157

'''
    OpenCV人脸级联分类器训练工具,用于生成Face recognition所需的xml文件
'''


import os.path
import cv2
import numpy
from config import HUMAN_FACE_IMAGES, HUMAN_FACE_CLASSIFIER, HUMAN_FACE_MODEL


class humanFace_training:
    @staticmethod
    def training():
        """
        Human Face training
        Silent execution in the background in threaded mode
        :return: No Return
        """
        try:
            labels = []
            faces = []
            face_detector = cv2.CascadeClassifier(HUMAN_FACE_CLASSIFIER)
            for folder in os.listdir(HUMAN_FACE_IMAGES):
                path = os.path.join(HUMAN_FACE_IMAGES, folder)
                files = os.listdir(path)
                for file in files:
                    imagePath = os.path.join(path, file)
                    im = cv2.imread(imagePath)
                    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    face = face_detector.detectMultiScale(grey)
                    for x, y, w, h in face:
                        labels.append(int(folder))
                        faces.append(grey[y:y + h, x:x + w])
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, numpy.array(labels))
            recognizer.write(HUMAN_FACE_MODEL)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    trainging = humanFace_training()
    trainging.training()