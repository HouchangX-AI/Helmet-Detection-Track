import numpy as np

from kcf import KcfFilter

class Instance(object):
    def __init__(self, config, video_helper, frame):
        #each history entry is an array with [frame_id, tag, bbx_left, bbx_right, bbx_up, bbx_down]
        self.history = []
        self.history_size = config.HISTORY_SIZE = 10


        self.num_misses = 0     # num of missed assignments
        self.max_misses = config.MAX_NUM_MISSING_PERMISSION
        self.has_match = False
        self.delete_duplicate = False
        self.delete_still = False
        self.delete_singular = False
        self.num_of_still = 0  # num of detector num

        self.kcf = KcfFilter(video_helper, frame)    # video_helper here can provide us the fps

        # this color is for bbx (color assigned to this instance itself）
        color = np.random.randint(0, 255, (1, 3))[0]
        self.color = [int(color[0]), int(color[1]), int(color[2])]  # color needs to be a tuple

        # this color is for central point
        # because we need to fade the color
        self.center_color = []
        self.center_color.append([int(color[0]), int(color[1]), int(color[2])])

        self.predicted_next_bbx = None

        self.num_established = 1       # num of sequential detections before we consider it a track

        self.COLOR_FADING_PARAM = config.COLOR_FADING_PARAM

        # we still need to change it later
        self.speed = 0
        self.direction = 0      # degree of velocity direction with [1, 0]

        self.still_history = 0

    # def add_to_track(self, det):
    def add_to_track(self, tag, bbx , frame):
        # Same function with correct_track
        # Just make it more clear when it is the case that this particular detection is a new one and need to be
        # tracked later.

        # since here we are assured that this instance is detected, so it's not missed
        #self.num_misses = 0

        # use measured data to correct the  filter
        # still a list [x_left, x_right, y_up, y_bottom]
        corrected_bbx = self.kcf.correct(bbx, frame)

        # history: [tag, bbx_left, bbx_right, bbx_up, bbx_down, [color_b, color_g, color_r]]
        new_history = [tag,
                       corrected_bbx[0], corrected_bbx[1], corrected_bbx[2], corrected_bbx[3],
                       [self.color[0], self.color[1], self.color[2]]]

        if (len(self.history) == 0):

            self.history.append(new_history)
        else:
            for i in range(len(self.history)):
                for c in range(3):
                    temp = self.history[i][5][c]
                    self.history[i][5][c] = int(
                        ((self.COLOR_FADING_PARAM - 1) / self.COLOR_FADING_PARAM) * temp)
                    if self.history[i][5][c] < 0:
                        self.history[i][5][c] = 0
            self.history.insert(0, new_history)
            # we need to cut if we set
            if len(self.history) == self.COLOR_FADING_PARAM - 1:
                del self.history[-1]
        self.num_of_still += 1

    def add_to_track_with_no_correction(self, tag, bbx, frame):
        new_history = [tag,
                       bbx[0], bbx[1], bbx[2], bbx[3],
                       [self.color[0], self.color[1], self.color[2]]]
        for i in range(len(self.history)):
            for c in range(3):
                temp = self.history[i][5][c]
                self.history[i][5][c] = int(
                    ((self.COLOR_FADING_PARAM - 1) / self.COLOR_FADING_PARAM) * temp)
                if self.history[i][5][c] < 0:
                    self.history[i][5][c] = 0
        self.history.insert(0, new_history)
        # we need to cut if we set
        if len(self.history) == self.COLOR_FADING_PARAM - 1:
            del self.history[-1]

    def correct_track(self, det, frame):
        # Same function with add_to_track.
        # Just make it more clear when it is the case that we have a track and we need to correct that after we
        # have the measurement.
        # det: {'tag' : [bbx_left, bbx_right, bbx_up, bbx_bottom]}
        tag = list(det.keys())[0]
        bbx = det[tag]
        self.add_to_track(tag, bbx, frame)

    def get_predicted_bbx(self,frame):
        # get a prediction
        return self.kcf.get_predicted_bbx(frame)

    def get_latest_bbx(self):
        if len(self.history) == 0:
            return []
        res = self.history[0]
        last_bbx = [res[1], res[2], res[3], res[4]]
        return last_bbx

    def get_ith_bbx(self, i):
        # if input i is int type
        if isinstance(i, int):
            if (i > 0 and i < len(self.history)) or (i < 0 and -i <= len(self.history)):
                res = self.history[i]
                ith_bbx = [res[1], res[2], res[3], res[4]]
                return ith_bbx

        # if input i is other type
        else:
            print("Wrong type!")
            return []

    def get_first_bbx(self):
        return self.get_ith_bbx(-1)

    def get_latest_record(self):
        if len(self.history) == 0:
            return []
        return self.history[0]

    def get_age(self):
        return len(self.history)





