import math
import numpy as np
import cv2

def get_area_from_coord(x1, x2, y1, y2):
    if x2 < x1 or y2 < y1:
        print("Wrong in GET AREA")
        exit()
    return (x2 - x1 + 1.0) * (y2 - y1 + 1.0)

def get_area_from_bbx(bbx):
    # bbx = [x_left, x_right, y_top, y_bottom]
    x1 = bbx[0]
    x2 = bbx[1]
    y1 = bbx[2]
    y2 = bbx[3]
    return get_area_from_coord(x1, x2, y1, y2)

def get_mask_area_in_img(img):
    return cv2.countNonZero(img)

def get_wh_ratio_from_coord(x1, x2, y1, y2):
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    _max = max(w, h)
    _min = min(w, h)
    return _max * 1.0 / _min

def vector2d_dis(v1, v2):
    return math.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2)

def get_centroid_from_bbx(bbx):
    x_left, x_right, y_top, y_bottom = bbx
    w = x_right - x_left + 1
    h = y_bottom - y_top + 1
    xc = x_left + w / 2
    yc = y_top + h / 2
    return [xc, yc]

def dist_btwn_bbx_centroids(bbx1, bbx2):
    # when it's from kalman filter, usually bbx1 = predicted_bbx, bbx2 = detected_bbx
    c1 = get_centroid_from_bbx(bbx1)  # 获取中心点
    c2 = get_centroid_from_bbx(bbx2)
    return vector2d_dis(c1, c2)

def check_instance_identical(instance1, instance2, threshold):
    bbx1 = instance1.get_latest_bbx()
    bbx2 = instance2.get_latest_bbx()
    dist = np.linalg.norm(np.array(bbx1) - np.array(bbx2))
    return dist < threshold

def check_instance_identical_by_iou(instance1, instance2, iou_threshold):
    bbx1 = instance1.get_latest_bbx()
    bbx2 = instance2.get_latest_bbx()
    iou = get_iou(bbx1, bbx2)
    if iou >= iou_threshold:
        return True
    else:
        return False

def check_bbxes_identical_by_ios(bbx_ins, bbx_det, ios_threshold):
    ios = get_ios(bbx_det, bbx_ins)
    if ios >= ios_threshold:
        return True
    else:
        return False

def check_blob_identical_by_ios(blob_ins, blob, ios_threshold):
    ios = get_ios(blob, blob_ins)
    if ios >= ios_threshold:
        return True
    else:
        return False

def get_iou(bbx1, bbx2):
    # bbx = [left, right, top, bottom]

    # if there's no intersection
    if bbx1[0] > bbx2[1] \
            or bbx1[1] < bbx2[0] \
            or bbx1[2] > bbx2[3] \
            or bbx1[3] < bbx2[2]:
        return 0

    x_min = max(bbx1[0], bbx2[0])
    x_max = min(bbx1[1], bbx2[1])
    y_min = max(bbx1[2], bbx2[2])
    y_max = min(bbx1[3], bbx2[3])

    # area of intersection rectangle
    inter_area = (x_max - x_min + 1.0) * (y_max - y_min + 1.0)

    # area of two bbxes
    bbx1_area = get_area_from_bbx(bbx1)
    bbx2_area = get_area_from_bbx(bbx2)

    # get iou
    iou = inter_area / float(bbx1_area + bbx2_area - inter_area)

    return iou

def get_ios(bbx1, bbx2):
    # intersection over itSelf
    # we return ios for the first argument

    # if there's no intersection
    if bbx1[0] > bbx2[1] \
            or bbx1[1] < bbx2[0] \
            or bbx1[2] > bbx2[3] \
            or bbx1[3] < bbx2[2]:
        return 0

    x_min = max(bbx1[0], bbx2[0])
    x_max = min(bbx1[1], bbx2[1])
    y_min = max(bbx1[2], bbx2[2])
    y_max = min(bbx1[3], bbx2[3])

    # area of intersection rectangle
    inter_area = (x_max - x_min + 1.0) * (y_max - y_min + 1.0)

    # area of bbx1
    bbx1_area = get_area_from_bbx(bbx1)

    # get ios
    ios = inter_area * 1.0 / bbx1_area

    return ios

def get_sum_still(bbx1, bbx2):
    sum = 0
    size1 = len(bbx1)
    size2 = len(bbx2)
    if size1 != size2:
        print("Wrong size in GET_SUM_STILL'")
        exit(0)
    for i in range(0, size1):
        sum += abs(bbx1[i] - bbx2[i])
    return sum

def get_vector_from_two_points(p1, p2):
    # p1: [x1, y1]
    # p2: [x2, y2]
    vec = [p1[0] - p2[0], p1[1] - p2[1]]
    return vec

def get_angle_from_two_vectors(vec1, vec2):
    vec = np.dot(np.array(vec1), np.array(vec2)) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    rad = np.arccos(np.clip(vec, -1, 1))
    deg = rad * 180 / math.pi
    return deg

def get_maxiou_id(id_iou):
    iou=[]
    for ll in id_iou:
        iou.append(ll[1])
    if len(iou) !=0:
        return(np.argmax(iou))
    else :
        return None
