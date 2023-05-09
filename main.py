#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @ Project: Face recognition 
# @ File: main.py
# @ Time: 18/2/2023 上午11:15
# @ Author: hz157
# @ Github: https://github.com/hz157
from hunman import HumanFace

if __name__ == '__main__':
    h = HumanFace(camera=0)
    h.lunchCamera()
    h.recognize()
