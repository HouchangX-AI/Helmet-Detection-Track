"""
This class aims to be the controller of all objects detected in a frame.
Mainly we are focusing on
"""

from instance import Instance
import util

import numpy as np
import sys
import cv2

# Munkres Alogrithm:
# Instructions and C# version can be found here: http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
# while in Python we can solve the problem by using method imported as below.
# Descriptions for Python version can be found in
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
from scipy.optimize import linear_sum_assignment 

class MultipleObjectController(object):
    def __init__(self, config, video_helper):
        self.instances = []
        self.instances_blob = []
        self.config = config
        self.video_helper = video_helper

    def update(self, detections, frame_id, frame):
        # In this mode, we update our tracking by using detection and prediction results.
        # Specific explanations are presented with the function.
        self.assign_detections_to_tracks(detections, frame_id, frame)

    def update_without_detection(self, frame_id, frame):
        # In this mode, we still need to predict next tracking by using historical info
        # though we don't have the detection
        #
        # step 1. update bbx by prediction
        for instance in self.instances:
            bbx = instance.get_predicted_bbx(frame)
            tag = instance.get_latest_record()[0]
            instance.add_to_track_with_no_correction(tag, bbx,frame)
            #instance.add_to_track(tag, bbx)
            instance.num_misses += 1

        # step 2. remove dead bboxes
        self.remove_dead_instances()

    def update_still(self, frame_id, frame):
        # In this mode, we remain current status when we need to predict because
        # sometimes, especially for some circumstances, video quality is extremely bad.
        # It's prone to loss our tracking.
        # In order to promote the effect of tracking, we only update our tracking whenever we need to detect
        # step 1. update bbx by doing nothing
        for instance in self.instances:
            bbx = instance.get_latest_bbx()
            tag = instance.get_latest_record()[0]
            instance.add_to_track_with_no_correction(tag, bbx, frame)
            instance.num_misses += 1
        # step 2. remove dead bboxes
        self.remove_dead_instances()

    def assign_detections_to_tracks(self, detections, frame_id, frame):
        """
        This function aims to assign detections to their corresponding tracks by minimizing the cost matrix using
        Munkres Algorithm.

        Args:
            param 1. detections: [{'tag1' : [bbx1]}, {'tag2' : [bbx2]}, ..., {'tagn' : [bbxn]}]
                     where bbx = [bbx_left, bbx_right, bbx_up, bbx_down]
            param 2. frame_id: current frame ID starting from 0
        """
        """ Step 0: got measurements (detections) """


        """ Step 1: get predicted states """
        # find distance from all tracked instances to all detected boxes
        # if there are no instances, then all detections are new tracks
        if (len(self.instances)) == 0:
            for det in detections:
                # det: {'tag' : [bbx_left, bbx_right, bbx_up, bbx_bottom]}
                instance = Instance(self.config, self.video_helper, frame)
                tag = list(det.keys())[0]
                bbx = det[tag]
                instance.add_to_track(tag, bbx, frame)
                self.instances.append(instance)
            return True

        """ Step 2: assign detections to correspondiong instances """
        # iou match between detected bboxes and tracked bboxes
        track_det_iou, det_track_iou = {}, {}
        for t, instance in enumerate(self.instances):
            if not track_det_iou.__contains__(t):
                track_det_iou[t] = [] # creat key
            predicted_bbx = instance.get_predicted_bbx(frame) # Here, by using KCF, we predict an bbx for each instance
            for d, det in enumerate(detections):
                if not det_track_iou.__contains__(d):
                    det_track_iou[d] = []
                detected_bbx = list(det.values())[0]
                iou = util.get_iou(predicted_bbx, detected_bbx) # get IOU between all detected bboxes and tracked bboxes
                track_det_iou[t].append([d, iou])
                det_track_iou[d].append([t, iou])

        # set all tracked instances as unassigned
        for instance in self.instances:
            instance.has_match = False

        assigned_instances, assigned_detections = [], []
        for i, id_iou in track_det_iou.items():
            match_detid = util.get_maxiou_id(id_iou)
            if match_detid != None:
               match_trackid = util.get_maxiou_id(det_track_iou[match_detid])
               if match_trackid == i:  # match
                    assigned_instances.append(match_trackid)
                    assigned_detections.append(match_detid)


        """ Step 3: correct an instance if it's matched by correponsding instance and detection """
        assigned_detection_id = []
        if assigned_instances != None and assigned_detections != None :  # sys.maxsize:

            for idx, instance_id in enumerate(assigned_instances):
                detection_id = assigned_detections[idx]
            # if assignment for this instance and detection is sys.maxsize, discard it
            # record this detection is assigned
                assigned_detection_id.append(detection_id)
                # record this tracked instance which is assigned
                self.instances[instance_id].has_match = True
                # correct states by using kalman filter
                self.instances[instance_id].correct_track(detections[detection_id], frame)
                # means this instance is detected
                self.instances[instance_id].num_misses = 0

        """ Step 4: remove instances which should be """
        # keep track of how many times a track has gone unassigned
        for instance in self.instances:
            if instance.has_match is False:
                instance.num_misses += 1
        # The function shown below can only remove those instances which has already been
        # added to tracks but CAN NOT remove detected bbx which has a huge IOU with
        # existed tracks. So we need another remove function to dual with that
        self.remove_dead_instances()

        """ Step 5: create new instances to track if no detections are matched """
        # get unassigned detection ids
        unassigned_detection_id = list(set(range(0, len(detections))) - set(assigned_detection_id))
        for idx in range(0, len(detections)):
            if idx in unassigned_detection_id:
                # det: {'tag' : [bbx_left, bbx_right, bbx_up, bbx_bottom]}
                tag = list(detections[idx].keys())[0]
                bbx = detections[idx][tag]
                # then we need to confirm whether the detection is a good one
                if self.is_good_detection(bbx):
                    instance = Instance(self.config, self.video_helper, frame)
                    instance.add_to_track(tag, bbx,frame)
                    self.instances.append(instance)

    def remove_dead_instances(self):
        # sometimes tracked instances may be duplicated by chances. so we need to delete replications
        self.delete_duplicate_tracks()
        # decide which track is still
        self.delete_still_tracks()
        # decide which track is singular
        self.delete_singular_tracks()
        # remain some instances
        self.instances = [instance for instance in self.instances if (
            # cond 1. this instance hasn't been assigned to a detection for a long time
            instance.num_misses <= self.config.MAX_NUM_MISSING_PERMISSION
            and
            # cond 2. remove still instances
            instance.delete_still is False     # if it's prediction mode, be careful it may delete them
            and
            # cond 3. remain non-duplicated instances/tracks
            instance.delete_duplicate is False
            # cond 4. remain non-singular instances
            and
            instance.delete_singular is False
        )]

    def delete_still_tracks(self):
        # Sometimes we may loss our tracking due to the high speed of objects or when objects are
        # moving to the edge. Bboxes will remain the same for several frames. We need to remove them
        for instance in self.instances:
            if len(instance.history) > self.config.NUM_DELETE_STILL:
                still_counter = 0
                for i in range(1, self.config.NUM_DELETE_STILL + 1):
                    bbx_i = instance.get_ith_bbx(-i)        # the last bbx
                    bbx_i_1 = instance.get_ith_bbx(-i - 1)  # the bbx before last one
                    sum_still = util.get_sum_still(bbx_i, bbx_i_1)
                    if sum_still == 0:
                        still_counter += 1
                # if still_counter >= self.config.NUM_DELETE_STILL:#########################
                #     instance.delete_still = True
                # else:
                #     instance.delete_still = False

    def delete_singular_tracks(self):
        # Sometimes we need to delete those tracks which are singular (too tiny, too large or too...slim)
        # in order we can get a better effect of tracking
        for instance in self.instances:
            # get current bbx： [left, right, top, bottom]
            curr_bbx = instance.get_latest_bbx()
            left = curr_bbx[0]
            right = curr_bbx[1]
            top = curr_bbx[2]
            bottom = curr_bbx[3]

            area = util.get_area_from_coord(left, right, top, bottom)
            wh_ratio = util.get_wh_ratio_from_coord(left, right, top, bottom)
            # if area is too big (e.g crowd)
            if area > self.config.AREA_MAXIMUM:
                instance.delete_singular = True
            elif area < self.config.AREA_MINIMUM:
                instance.delete_singular = True
            elif wh_ratio > self.config.WH_RATIO_THRE:
                instance.delete_singular = True
            else:
                instance.delete_singular = False

    def delete_duplicate_tracks(self):
        for i in range(len(self.instances)):
            ins1 = self.instances[i]
            for j in range(len(self.instances)):
                if i == j:
                    continue
                ins2 = self.instances[j]
                # if we think they are identical
                if util.check_instance_identical_by_iou(
                        ins1,
                        ins2,
                        self.config.INSTANCE_IDENTICAL_IOU_THRESHOLD):
                    # then we need to know which one is younger
                    if (ins1.get_age() > ins2.get_age()):
                        ins2.delete_duplicate = True
                        ins1.delete_duplicate = False
                    else:
                        ins1.delete_duplicate = True
                        ins2.delete_duplicate = False

    def is_good_detection(self, bbx):
        #
        for instance in self.instances:
            if util.check_bbxes_identical_by_ios(
                instance.get_latest_bbx(),
                bbx,
                self.config.BBXES_IDENTICAL_IOS_TRHESHOLD
            ):
                return False
        return True

