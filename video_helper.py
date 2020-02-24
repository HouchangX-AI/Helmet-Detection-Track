import cv2
import numpy as np
from PIL import Image
class VideoHelper(object):
    """
    This class will help us to duel with operations related with videos
    such as open/close, read/write video and also we can use this helper
    to get attributes of the video
    """

    def __init__(self, config):
        # video in
        self.video_in = cv2.VideoCapture()
        self.video_in.open(config.VID_NAME)

        # video attributes
        self.frame_width = int(self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH))       # 3: cv2.CAP_PROP_FRAME_WIDTH
        self.frame_height = int(self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))     # 4: cv2.CAP_PROP_FRAME_HEIGHT
        self.frame_fps = int(self.video_in.get(cv2.CAP_PROP_FPS))                 # 5: cv2.CAP_PROP_FPS

        # video output
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.video_out = cv2.VideoWriter(config.VID_SAVING_NAME, fourcc, self.frame_fps, (self.frame_width, self.frame_height))
        self.video_blob_out = cv2.VideoWriter(config.VID_SAVING_BLOB_NAME, fourcc, self.frame_fps, (config.BACK_RESIZE_WIDTH, config.BACK_RESIZE_HEIGHT))

        # cut the video at this num
        self.finish_frame_num = config.FINISH_CUT_FRAME  # in case the video is too long

    def not_finished(self, cur_frame):
        """
        This function is used to check whether we finish reading video file or not

        Args:
            param1: cur_frame: # of current frame
        Return:
            bool type to check whether we can stop reading the video file or not
        """
        if self.video_in.isOpened():
            if self.finish_frame_num == 0:
                return True
            if cur_frame < self.finish_frame_num:
                return True
            else:
                return False
        else:
            print("Video is NOT opened!")
            return False

    def get_frame(self):
        """
        This function is used to get current frame from the video file

        Return:
            frame: raw frame
            frame_show: use this to show the result
        """
        ret, frame = self.video_in.read()
        # frame is none
        if ret != True:
            print("That's all!")
            exit()
        frame_show = frame
        frame_PIL = Image.fromarray(frame.astype('uint8')).convert('RGB')
        return frame_PIL, frame, frame_show

    def write_video(self, img):
        self.video_out.write(img)


    def end(self):
        self.video_in.release()
        self.video_out.release()
        self.video_blob_out.release()



