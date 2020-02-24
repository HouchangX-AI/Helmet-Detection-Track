class Configs(object):
    """
    Configs(Routes) for the whole detection/tracking system
    """

    def __init__(self):
        """ Sever or Local """
        self.S_or_L = 'S'    # running on sever or local environment. 's' or 'S': sever, 'l' or 'L': local
        self.VID_NAME = '1582436108570269.mp4'
        self.VID_SAVING_NAME = 'save_video'
        self.VID_SAVING_BLOB_NAME = 'VID_SAVING_BLOB_NAME'
        """
        Detector
        """
        self.AREA_MAXIMUM = 10000000    # Instance which area is bigger than this number will not be regarded as an
                                    # independent instance
        self.AREA_MINIMUM = 400     # Instance which area is smaller than this number will not be regarded as a
                                    # valid instance
        self.WH_RATIO_THRE = 3      # max(width, height) / min(width, height)
        self.COOLING_FRAME = 100    # For some reasons, yolo is not good enough to detect blurred objects. We guess blob
                                    # analysis may be a better choice thus we need a time to build up our background.
        self.BACK_HISTORY_FRAME = 300       # We use these frames to build up background model in real sense.
                                            # It should be less than COOLING_FRAME
        self.BACK_THRESHOLD = 100           # Threshold of background model. The higher, the more inertial.
        self.BACK_IF_DETECT_SHADOW = False  # Whether to detect shadow
        self.BACK_RESIZE_BORDER = 480
        self.BACK_RESIZE_HEIGHT = 240
        self.BACK_RESIZE_WIDTH = 320

        self.USE_RE_MODEL = False

        """
        # Multiple Object Controller Parameters
        """
        # it's better to set to ratio of bboxes cause smaller ones means shorter dist and vice versa.
        self.MAX_PIXELS_DIST_BETWEEN_PREDICTED_AND_DETECTED = 45
        # detect per NUM_JUMP_FRAMES frames. D, N, N, N, D, N, ...
        self.NUM_JUMP_FRAMES = 5
        self.VELOCITY_DIRECTION_SEPARATOR = 90
        """
        # Instance Parameters
        """
        # An instance will be deleted if we cannot detected it after this number
        self.MAX_NUM_MISSING_PERMISSION = self.NUM_JUMP_FRAMES+1
        # An instance will be showed if we  detected it more than this number
        self.MIN_CONTINUE_DETECTOR = 2
        # We save historical frames for each instance within this number
        self.HISTORY_SIZE = 20
        # we need to do get rid of those identical instances generated
        # but it's not good in real experiments
        self.INSTANCE_IDENTICAL_THRESHOLD = 50
        # so we set iou threshold to do the same thing
        self.INSTANCE_IDENTICAL_IOU_THRESHOLD = 0.3
        # we also set ios (inter over self) threshold
        # for conditions where two instances' areas are different tremendously
        self.BBXES_IDENTICAL_IOS_TRHESHOLD = 0.35
        # If bbx remains same within this # of frames, we delete it.
        self.NUM_DELETE_STILL = self.NUM_JUMP_FRAMES + 1


        """
        Visualizer Parameters
        """
        # By using CPU, we can use this flag to decide whether to show results directly or not
        self.to_show = False
        self.show_path = 'vis/'   # save images path
        # If the video is too long, we gonna stop it at this frame
        self.FINISH_CUT_FRAME = 0
        self.COLOR_FADING_PARAM = self.HISTORY_SIZE
        self.SHOW_FRAME_ID = True

        self.SHOW_COLLISION_THRE = 50

        self.SHOW_TRACKS = True








