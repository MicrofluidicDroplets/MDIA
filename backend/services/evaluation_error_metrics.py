import json
import pandas as pd
from tqdm import tqdm


def box_std(box):
    """
    矩形框格式转换,将输入的矩形框坐标转换为左上角和右下角的坐标，方便后续计算 格式为[x1,y1,x2,y2]
    :param box:
    :return:
    """
    # 检查矩形框格式
    if len(box) != 5 and len(box) % 2 == 1:
        raise ValueError('box格式错误')
    if len(box) == 2:
        if len(box[0]) == 1:
            raise ValueError('box格式错误')
        box = [box[0][0], box[0][1], box[1][0], box[1][1]]
    # 矩形框格式转换
    if box[0] > box[2]:
        box[0], box[2] = box[2], box[0]
    if box[1] > box[3]:
        box[1], box[3] = box[3], box[1]
    return box


def long_error(box_1, box_2):
    """
        计算两个矩形框的中心点误差
        :param box_1: 程序识别的矩形框
        :param box_2: 手动标记的矩形框
    """
    box1 = box_std(box_1)
    box2 = box_std(box_2)
    error1 = (box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2
    error2 = (box1[2] - box2[2]) ** 2 + (box1[3] - box2[3]) ** 2
    error3 = (box2[0] - box2[2]) ** 2 + (box2[1] - box2[3]) ** 2
    if error3 == 0:
        error3 = 1
    error_long = abs((error1 ** 0.5 + error2 ** 0.5) / (error3 ** 0.5))
    return error_long


def area_error(box_1, box_2):
    """
    计算两个矩形框的面积误差
    :param box_1: 矩形框1
    :param box_2: 矩形框2
    :return: 以box1为基准的面积误差
    """
    box1 = box_std(box_1)
    box2 = box_std(box_2)
    area1 = abs(box1[2] - box1[0]) * abs(box1[3] - box1[1])
    area2 = abs(box2[2] - box2[0]) * abs(box2[3] - box2[1])
    area_error_ = abs(area1 - area2) / area1
    return area_error_


def Short_long_ratio(box_1):
    """
    计算短边长边比
    :param box_1: 矩形框
    :return: 短边长边比
    """
    box1 = box_std(box_1)
    long = abs(box1[2] - box1[0])
    width = abs(box1[3] - box1[1])
    if width == 0:
        width = 1
    if long == 0:
        long = 1
    short_long_ratio = min(long, width) / max(long, width)
    return short_long_ratio


def Short_long_filtration(boxes, threshold=0.8):
    """
    短边长边比过滤
    :param boxes: 矩形框列表
    :param threshold: 阈值
    :return: 过滤后的矩形框列表
    """
    new_boxes = []
    for box in boxes:
        if threshold < Short_long_ratio(box) <= 1:
            new_boxes.append(box)
    return new_boxes


def calculate_iou(box1, box2):
    """
    计算交并比
    Calculate intersection over union
    :param box1: bounding box 1 [x1, y1, x2, y2]
    :param box2: bounding box 2 [x1, y1, x2, y2]
    :return: intersection over union
    """
    box1 = box_std(box1)
    box2 = box_std(box2)
    # Calculate the coordinates of the rectangles
    x1_tl = box1[0]
    x2_tl = box2[0]
    x1_br = box1[2]
    x2_br = box2[2]
    y1_tl = box1[1]
    y2_tl = box2[1]
    y1_br = box1[3]
    y2_br = box2[3]
    # Calculate the area of rectangles
    area_box1 = max(0, x1_br - x1_tl ) * max(0, y1_br - y1_tl)
    area_box2 = max(0, x2_br - x2_tl ) * max(0, y2_br - y2_tl)
    # Calculate the coordinates of the intersection rectangle
    x1_tl_inter = max(x1_tl, x2_tl)
    y1_tl_inter = max(y1_tl, y2_tl)
    x2_br_inter = min(x1_br, x2_br)
    y2_br_inter = min(y1_br, y2_br)
    # Calculate the area of intersection rectangle
    area_intersection = max(0, x2_br_inter - x1_tl_inter ) * max(0, y2_br_inter - y1_tl_inter )
    # Calculate the area of union of two rectangles
    area_union = area_box1 + area_box2 - area_intersection
    # Calculate intersection over union
    iou = area_intersection / area_union
    return iou


def nms(bounding_boxes, intersection_threshold=0.5):
    """
    Non-maximum suppression
    # 过滤掉重叠度高的边界框
    :param bounding_boxes: list of bounding boxes   [x1, y1, x2, y2, score]
    :param intersection_threshold: intersection threshold # 交集阈值
    :return: list of bounding boxes after nms # nms后的边界框列表
    """
    if not bounding_boxes:
        return []
    # Sort the bounding boxes by their confidence scores in descending order
    sorted_bounding_boxes = sorted(bounding_boxes, key=lambda x: x[4], reverse=True)
    new_bounding_boxes = [sorted_bounding_boxes[0]]
    del sorted_bounding_boxes[0]
    for box in sorted_bounding_boxes:
        flag = 1
        for new_box in new_bounding_boxes:
            if calculate_iou(box, new_box) > intersection_threshold:
                flag = 0
                break
        if flag:
            new_bounding_boxes.append(box)
    return new_bounding_boxes


def filter_bounding_boxes(bounding_boxes, min_score=0.3, nms_flag=True, nms_threshold=0.5,short_long_ratio_flag=True,
                          short_long_ratio_threshold=0.8):
    """
    过滤掉置信度低的边界框
    :param bounding_boxes: 边界框列表
    :param min_score: 置信度阈值
    :param nms_flag: 是否进行nms
    :param nms_threshold: nms阈值
    :param short_long_ratio_flag: 是否进行短边长边比过滤
    :param short_long_ratio_threshold:  短边长边比阈值
    :return: bounding_boxes 过滤后的边界框列表
    """
    bounding_boxes = [bounding_box for bounding_box in bounding_boxes if bounding_box[4] > min_score]
    if nms_flag:
        bounding_boxes = nms(bounding_boxes, nms_threshold)
    if short_long_ratio_flag:
        bounding_boxes = Short_long_filtration(bounding_boxes, short_long_ratio_threshold)
    return bounding_boxes


def read_labelme(file):
    """
    Read labelme json file 为json2coco 输出的coco格式的json文件
    imgs_info: [img_id, img_name, img_height, img_width] 图片信息 包括图片id, 图片名称, 图片高度, 图片宽度
    labelme_data: [img_id, img_image_id, img_category_id, img_segmentation] 标注信息 包括标注id, 图片id, 类别id, 分割信息
    :param file: json file
    :return: imgs_info, labelme_data
    """
    labelme_data = []
    imgs_info = []
    with open(file, "r") as in_file:
        json_dict = json.load(in_file)
        for imgs in tqdm(json_dict["images"], desc='读取图片信息', colour='green'):
            img_height = imgs['height']
            img_width = imgs['width']
            img_id = imgs['id']
            img_name = imgs['file_name']
            imgs_info.append([img_id, img_name, img_height, img_width])

        for annotations in tqdm(json_dict["annotations"], desc='读取标注信息', colour='green'):
            img_id = annotations['id']
            img_image_id = annotations['image_id']
            img_category_id = annotations['category_id']
            img_segmentation = annotations['segmentation']
            img_segmentation = img_segmentation[0]

            labelme_data.append([img_id, img_image_id, img_category_id, img_segmentation])
    return imgs_info, labelme_data


def read_pkl(file, nms_flag=True, nms_threshold=0.5,short_long_ratio_flag=True,short_long_ratio_threshold=0.8):
    """
    读取pkl文件
    返回格式为[image_id, category_id, bounding_box]
    :param file: pkl file
    :param nms_flag: 是否进行nms
    :param nms_threshold: nms阈值
    :param short_long_ratio_flag: 是否进行短边长边比过滤
    :param short_long_ratio_threshold: 短边长边比阈值
    :return: pkl_result [image_id, category_id, bounding_box]
    """
    data = pd.read_pickle(file)
    pkl_result = []
    for i in range(len(data)):
        image_id = i
        for j in range(len(data[i])):
            category_id = j + 1
            bounding_boxes = filter_bounding_boxes(data[i][j], nms_flag=nms_flag, nms_threshold=nms_threshold,
                                                   short_long_ratio_flag=short_long_ratio_flag,
                                                   short_long_ratio_threshold=short_long_ratio_threshold)
            for bounding_box in bounding_boxes:
                pkl_result.append([image_id, category_id, bounding_box])
    return pkl_result


def calculate_error(labelme_data, pkl_result):
    """
    计算误差
    :param labelme_data: 标注信息
    :param pkl_result: 预测信息
    :return: error_result [image_id, category_id, bounding_box, best_gt_box, best_iou, long_error_value, area_error_value]
    [图片id, 类别id, 预测框, 最佳匹配标注框, 最佳匹配iou, 中心点误差, 面积误差]
    """
    error_result = []
    for pkl in tqdm(pkl_result, desc='计算误差', colour='green'):
        image_id = pkl[0]
        category_id = int(pkl[1]) + 1
        bounding_box = pkl[2]
        best_iou = 0
        best_gt_box = None
        for labelme in labelme_data:
            if labelme[1] == image_id and labelme[2] == category_id:
                iou = calculate_iou(bounding_box, labelme[3])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_box = labelme[3]
        if best_iou > 0.5:
            long_error_value = long_error(bounding_box, best_gt_box)
            area_error_value = area_error(bounding_box, best_gt_box)
            error_result.append(
                [image_id, category_id, bounding_box, best_gt_box, best_iou, long_error_value, area_error_value])
        else:
            error_result.append([image_id, category_id, bounding_box, best_gt_box, None, None, None])
    return error_result
