"""
2 usages:
a. To smooth noisy detection results
b. To accelerate the whole procedure by merely using predict with some frames
   rather than detecting per frame
"""

import cv2
import numpy as np
import math
from config import Configs
from video_helper import VideoHelper

from tracker import kcftracker

# First Trial:
# 6 element in a state: [xc, yc, vx, vy, w, h]
# 4 element in a measure: [zxc, zyc, zw, zh]
class KcfFilter(object):
    def __init__(self, video_helper, frame):
        self.first_run = True
        self.dynamParamsSize = 6
        self.measureParamsSize = 4
        self.kcf = kcftracker.KCFTracker(True, True, True)

    def correct(self, bbx, frame):
        # need to be numpy array
        # measurement is numpy array
        # bbx: x_left, x_right, y_up, y_bottom
        # need to convert to [xc, yc, w, h] first
        w = bbx[1] - bbx[0] + 1
        h = bbx[3] - bbx[2] + 1
        xc = int(bbx[0] + w / 2)
        yc = int(bbx[2] + h / 2)
        measurement = np.array([[xc, yc, w, h]], dtype = np.float32).T
        if self.first_run is True:
            self.kcf.init(measurement, frame)  #################
            #     statePre = np.array(
            #         [measurement[0], measurement[1], [0], [0], measurement[2], measurement[3]],
            #         dtype = np.float32
            # )
            ##self.first_run = False   #每次检测都重新初始化框
        corrected_res = self.kcf.update(frame)  ### correct(measurement).T[0]
        self.velocity = np.array([corrected_res[2], corrected_res[3]])

        # convert back to bbx form: x_left, x_right, y_up, y_bottom
        corrected_bbx = self.get_bbx_from_kcf_form(corrected_res)
        return corrected_bbx

    def get_predicted_bbx(self, frame):
        predicted_res = self.kcf.update(frame)#.T[0]
        predicted_bbx = self.get_bbx_from_kcf_form(predicted_res)
        return predicted_bbx

    def get_bbx_from_kcf_form(self, kcf_form):
        xc = kcf_form[0]
        yc = kcf_form[1]
        w = kcf_form[2]
        h = kcf_form[3]
        x_left = math.ceil(xc - w / 2.0)
        x_right = math.ceil(xc + w / 2.0) - 1
        y_up = math.ceil(yc - h / 2.0)
        y_bottom = math.ceil(yc + h / 2.0) - 1
        return [x_left, x_right, y_up, y_bottom]















