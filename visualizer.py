import cv2
import numpy as np

class Visualizer(object):
    """
    We can show result by using this class and get new frame_show
    which is the frame need to write to video
    """

    def __init__(self, configs):
        # to_show is a bool type variable which is the flag to tell us
        # whether we will show the result directly
        # Typically, if it's running in CPU, we set it to True while in GPU False
        self.to_show = configs.to_show
        self.config = configs

        self.start_collision = -1
        pass

    # version 1. just draw bbox for each frame
    def drawing_detects(self, img, outputs):
        if len(outputs) > 0:
            for output in outputs:
                # get current tag & bbx
                tag = list(output.keys())[0]
                bbx = output[tag]

                left = bbx[0]
                right = bbx[1]
                top = bbx[2]
                bottom = bbx[3]

                # get border
                bbx_width = right - left + 1
                bbx_height = bottom - top + 1
                border = bbx_height if bbx_width >= bbx_height else bbx_width

                # this color is for bbx (color assigned to this instance itself
                color = np.random.randint(0, 255, (1, 3))[0]
                self.color = [int(color[0]), int(color[1]), int(color[2])]  # color needs to be a tuple
                # draw corner of the bbx
                self.draw_corner(img, left, right, top, bottom, border, self.color)
                # draw tag
                # draw bbx
                self.draw_bbx(img, left, right, top, bottom, self.color)

    # version 2. draw tracking bboxes without temporal information
    def drawing_tracking(self, img, instances, frame_id, show_temporal_information):
        if len(instances) > 0 :
            for instance in instances:
                if instance.num_of_still >= self.config.MIN_CONTINUE_DETECTOR:
                    if not show_temporal_information:
                        # no need to show historical information (centers)
                        ins = instance.get_latest_record()
                        # [tag, bbx_left, bbx_right, bbx_up, bbx_bottom, color]
                        tag = ins[0]
                        left = ins[1]
                        right = ins[2]
                        top = ins[3]
                        bottom = ins[4]
                        color = ins[5]

                        center_x = int((left + right) / 2)
                        center_y = int((top + bottom) / 2)

                        # get border
                        bbx_width = right - left + 1
                        bbx_height = bottom - top + 1
                        border = bbx_height if bbx_width >= bbx_height else bbx_width

                        # draw tag
                        self.draw_tag(img, tag, left, right, top, bottom, border)

                        # draw bbx
                        self.draw_bbx(img, left, right, top, bottom, color)
                        # draw corner of the bbx
                        self.draw_corner(img, left, right, top, bottom, border, color)
                        # draw center
                        self.draw_center(img, center_x, center_y, color)
                        if self.config.SHOW_FRAME_ID:
                            # draw frame id
                            self.draw_frameid(img, str(frame_id))

                    else:
                        # show temporal centers
                        history = instance.history
                        ins = instance.get_latest_record()

                        tag = ins[0]
                        left = ins[1]
                        right = ins[2]
                        top = ins[3]
                        bottom = ins[4]
                        color = ins[5]

                        # get border
                        bbx_width = right - left + 1
                        bbx_height = bottom - top + 1
                        border = bbx_height if bbx_width >= bbx_height else bbx_width

                        # draw tag
                        self.draw_tag(img, tag, left, right, top, bottom, border)
                        # draw bbx
                        self.draw_bbx(img, left, right, top, bottom, color)
                        # draw corner of the bbx
                        self.draw_corner(img, left, right, top, bottom, border, color)
                        # draw temporal centers
                        for temp in history:
                            temp_left = temp[1]
                            temp_right = temp[2]
                            temp_top = temp[3]
                            temp_bottom = temp[4]
                            temp_cx = int((temp_left + temp_right) / 2)
                            temp_cy = int((temp_top + temp_bottom) / 2)
                            temp_color = temp[5]
                            self.draw_center(img, temp_cx, temp_cy, temp_color)
                        if self.config.SHOW_FRAME_ID:
                            # draw frame id
                            self.draw_frameid(img, str(frame_id))
        cv2.imwrite(self.config.show_path + str(frame_id) + '.jpg', img)



    def drawing_all(self, img, instances, frame_id, is_collision, show_temporal_information):
        if len(instances) > 0:
            for instance in instances:
                if not show_temporal_information:
                    # no need to show historical information (centers)
                    ins = instance.get_latest_record()
                    # [tag, bbx_left, bbx_right, bbx_up, bbx_bottom, color]
                    face = instance.face_id
                    emotion = instance.emotion
                    tag = face + '/' + emotion

                    left = ins[1]
                    right = ins[2]
                    top = ins[3]
                    bottom = ins[4]
                    color = ins[5]

                    center_x = int((left + right) / 2)
                    center_y = int((top + bottom) / 2)

                    # get border
                    bbx_width = right - left + 1
                    bbx_height = bottom - top + 1
                    border = bbx_height if bbx_width >= bbx_height else bbx_width

                    # draw tag
                    self.draw_tag(img, tag, left, right, top, bottom, border)
                    # draw bbx
                    self.draw_bbx(img, left, right, top, bottom, color)
                    # draw corner of the bbx
                    self.draw_corner(img, left, right, top, bottom, border, color)
                    # draw center
                    self.draw_center(img, center_x, center_y, color)
                    if self.config.SHOW_FRAME_ID:
                        # draw frame id
                        self.draw_frameid(img, str(frame_id))
                    if is_collision:
                        self.start_collision = frame_id
                    if self.start_collision != -1:
                        if frame_id - self.start_collision <= self.config.SHOW_COLLISION_THRE: # 50
                            self.draw_collision(img)
                else:
                    # show temporal centers
                    history = instance.history
                    ins = instance.get_latest_record()

                    face = instance.face_id
                    emotion = instance.emotion
                    tag = face + '/' + emotion
                    left = ins[1]
                    right = ins[2]
                    top = ins[3]
                    bottom = ins[4]
                    color = ins[5]

                    # get border
                    bbx_width = right - left + 1
                    bbx_height = bottom - top + 1
                    border = bbx_height if bbx_width >= bbx_height else bbx_width

                    # draw tag
                    self.draw_tag(img, tag, left, right, top, bottom, border)
                    # draw bbx
                    self.draw_bbx(img, left, right, top, bottom, color)
                    # draw corner of the bbx
                    self.draw_corner(img, left, right, top, bottom, border, color)
                    # draw temporal centers
                    for temp in history:
                        temp_left = temp[1]
                        temp_right = temp[2]
                        temp_top = temp[3]
                        temp_bottom = temp[4]
                        temp_cx = int((temp_left + temp_right) / 2)
                        temp_cy = int((temp_top + temp_bottom) / 2)
                        temp_color = temp[5]
                        self.draw_center(img, temp_cx, temp_cy, temp_color)
                    if self.config.SHOW_FRAME_ID:
                        # draw frame id
                        self.draw_frameid(img, str(frame_id))
                    if is_collision:
                        self.start_collision = frame_id
                    if self.start_collision != -1:
                        if frame_id - self.start_collision <= self.config.SHOW_COLLISION_THRE: # 50
                            self.draw_collision(img)

    def showing_tracking_blobs(self, img, blobs, frame_id, show_temporal_information):
        img_drawing = img.copy()
        if len(blobs) > 0:
            for blob in blobs:
                if not show_temporal_information:
                    # no need to show historical information (centers)
                    ins = blob.get_latest_record()
                    # [bbx_left, bbx_right, bbx_up, bbx_bottom, color]
                    left = ins[0]
                    right = ins[1]
                    top = ins[2]
                    bottom = ins[3]
                    color = ins[4]

                    center_x = int((left + right) / 2)
                    center_y = int((top + bottom) / 2)

                    # get border
                    bbx_width = right - left + 1
                    bbx_height = bottom - top + 1
                    border = bbx_height if bbx_width >= bbx_height else bbx_width

                    # draw bbx
                    self.draw_bbx(img_drawing, left, right, top, bottom, color)
                    # draw corner of the bbx
                    self.draw_corner(img_drawing, left, right, top, bottom, border, color)
                    # draw center
                    self.draw_center(img_drawing, center_x, center_y, color)
                else:
                    # show temporal centers
                    history = blob.history
                    ins = blob.get_latest_record()

                    left = ins[0]
                    right = ins[1]
                    top = ins[2]
                    bottom = ins[3]
                    color = ins[4]

                    # get border
                    bbx_width = right - left + 1
                    bbx_height = bottom - top + 1
                    border = bbx_height if bbx_width >= bbx_height else bbx_width


                    # draw bbx
                    self.draw_bbx(img_drawing, left, right, top, bottom, color)
                    # draw corner of the bbx
                    self.draw_corner(img_drawing, left, right, top, bottom, border, color)
                    # draw temporal centers
                    for temp in history:
                        temp_left = temp[0]
                        temp_right = temp[1]
                        temp_top = temp[2]
                        temp_bottom = temp[3]
                        temp_cx = int((temp_left + temp_right) / 2)
                        temp_cy = int((temp_top + temp_bottom) / 2)
                        temp_color = temp[4]
                        self.draw_center(img_drawing, temp_cx, temp_cy, temp_color)
                    if self.config.SHOW_FRAME_ID:
                        # draw frame id
                        self.draw_frameid(img, str(frame_id))
        return img_drawing

    def draw_corner(self, img, left, right, top, bottom, border, color):
        length = 2 if border / 5 <= 1 else border / 5
        left_end = int(left + length - 1)
        right_end = int(right - length + 1)
        top_end = int(top + length - 1)
        bottom_end = int(bottom - length + 1)
        line_color = (color[0], color[1], color[2])

        # draw left top corner    #(int(color[0]), int(color[1]), int(color[2]))
        cv2.line(img, (left, top), (left_end, top), line_color, 2, 8)
        cv2.line(img, (left, top), (left, top_end), line_color, 2, 8)

        # draw right top corner
        cv2.line(img, (right, top), (right_end, top), line_color, 2, 8)
        cv2.line(img, (right, top), (right, top_end), line_color, 2, 8)

        # draw left bottom corner
        cv2.line(img, (left, bottom), (left_end, bottom), line_color, 2, 8)
        cv2.line(img, (left, bottom), (left, bottom_end), line_color, 2, 8)

        # draw right bottom corner
        cv2.line(img, (right, bottom), (right_end, bottom), line_color, 2, 8)
        cv2.line(img, (right, bottom), (right, bottom_end), line_color, 2, 8)

    def draw_center(self, img, cx, cy, color):
        cv2.circle(img, (cx, cy), 2, color, cv2.FILLED)

    def draw_tag(self, img, tag, left, right, top, bottom, border): # img, tag, left, right, top, bottom, border
        # get text attribute
        textSize = cv2.getTextSize(tag, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        text_width = textSize[0][0]
        text_height = textSize[0][1]
        text_baseline = textSize[1]
        text_org = (left, top - text_baseline)
        text_bbx_left_top = (left, top - int(1.5 * text_baseline) - text_height)
        text_bbx_right_bottom = (left + text_width, top)

        # draw tag background
        cv2.rectangle(img,
                      text_bbx_left_top,
                      text_bbx_right_bottom,
                      (145, 145, 145),
                      cv2.FILLED)

        # put tag on
        cv2.putText(img,
                    tag,
                    text_org,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    True)

    def draw_bbx(self, img, left, right, top, bottom, color):
        cv2.rectangle(img, (left, top), (right, bottom), color, 1)

    def draw_frameid(self, img, id_str):
        cv2.putText(img,
                    id_str,
                    (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    True)










