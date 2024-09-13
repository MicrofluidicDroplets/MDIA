from evaluation_error_metrics import read_pkl, box_std, calculate_iou
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def calculate_diameter(bounding_box):
    """
    计算直径
    :param bounding_box:
    :return:
    """
    bounding_box = box_std(bounding_box)[:4]
    x1, y1, x2, y2 = bounding_box
    long = abs(x2 - x1)
    width = abs(y2 - y1)
    diameter = (long + width) / 2
    return diameter


def calculate_volume_ratio(bounding_boxes_inner, bounding_box_outer):
    """
    计算体积比
    :param bounding_boxes_inner: 包含多个核的列表
    :param bounding_box_outer: 外壳
    :return:
    """
    # 计算核的体积

    volume_inner = 0
    for bounding_box in bounding_boxes_inner:
        diameter = calculate_diameter(bounding_box)
        volume_inner += 4 / 3 * 3.14 * (diameter / 2) ** 3
    # 计算外壳的体积
    diameter_outer = calculate_diameter(bounding_box_outer)
    volume_outer = 4 / 3 * 3.14 * (diameter_outer / 2) ** 3
    volume_ratio = volume_inner / volume_outer
    return volume_ratio


def calculate_center(bounding_box):
    """
    计算中心点
    :param bounding_box:
    :return:
    """
    bounding_box = box_std(bounding_box)
    x1, y1, x2, y2 = bounding_box[:4]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def data_describe(data_pd, variable_name):
    """
    数据总结
    :param data_pd:
    :param variable_name: 类型名称
    :return:
    """
    # 从小到大排序
    data_pd = data_pd.sort_values(ascending=True, by=variable_name)
    data_pd = data_pd.reset_index(drop=True)
    # 计算是不是4的倍数，如果是，按第一种方法计算四分位数
    # 去除空值
    data_pd = data_pd.dropna()
    data_pd_desc = data_pd.describe()
    count = float(data_pd_desc.loc['count'][0])
    average = float(data_pd_desc.loc['mean'][0])
    std = float(data_pd_desc.loc['std'][0])
    variance = float(data_pd.var())
    min_ = float(data_pd_desc.loc['min'][0])
    P25 = float(data_pd_desc.loc['25%'][0])
    P50 = float(data_pd_desc.loc['50%'][0])
    P75 = float(data_pd_desc.loc['75%'][0])
    if len(data_pd) % 4 == 0:
        # 如果是4的倍数，则下分位数位于（n/4）和（n/4+1）的平均值
        P25 = float((data_pd.loc[int(len(data_pd) / 4 - 1)]) + data_pd.loc[int(len(data_pd) / 4)]) / 2
        P75 = float((data_pd.loc[int(len(data_pd) * 3 / 4 - 1)]) + data_pd.loc[int(len(data_pd) * 3 / 4)]) / 2
    if len(data_pd) % 4 != 0:
        # 如果不是4的倍数，则向上取整
        P25 = float(data_pd.loc[int(len(data_pd) / 4)])
        P75 = float(data_pd.loc[int(len(data_pd) * 3 / 4)])
    data_max = data_pd_desc.loc['max']
    data_max = float(data_max[0])
    Q1 = P25
    Q3 = P75
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier = []
    for i in data_pd[variable_name]:
        if i < lower_bound or i > upper_bound:
            outlier.append(i)
    # 将outlier转换为列表
    outlier = [round(float(i), 2) for i in outlier]
    if len(outlier) == 0:
        outlier_list = None
    else:
        # 将列表里的数字保留4位有效数字
        outlier_list = [round(float(i), 2) for i in outlier]

    Results = [[variable_name, count, average, std, variance, min_, P25, P50, P75, data_max, outlier_list]]
    Results = pd.DataFrame(Results, columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    return Results


def calculate_concentricity(bounding_box_inner, bounding_box_outer):
    """
    计算同心度
    :param bounding_box_inner:
    :param bounding_box_outer:
    :return:
    """
    center_inner = calculate_center(bounding_box_inner)
    center_outer = calculate_center(bounding_box_outer)
    distance = ((center_inner[0] - center_outer[0]) ** 2 + (center_inner[1] - center_outer[1]) ** 2) ** 0.5
    concentricity = 1 - distance / calculate_diameter(bounding_box_outer)
    return concentricity


def calculate_core_to_shell_ratio(bounding_box_inner, bounding_box_outer):
    """
    计算核壳比
    :param bounding_box_inner:
    :param bounding_box_outer:
    :return:
    """
    diameter_inner = calculate_diameter(bounding_box_inner)
    diameter_outer = calculate_diameter(bounding_box_outer)
    core_to_shell_ratio = diameter_inner / diameter_outer
    return core_to_shell_ratio


def match_core_shell(bounding_box_inner, bounding_box_outer):
    """
    # 计算交集，如果交集面积大于核的面积的0.9，则认为匹配成功
    :param bounding_box_inner:
    :param bounding_box_outer:
    :return:
    """
    bounding_box_inner = box_std(bounding_box_inner)
    bounding_box_outer = box_std(bounding_box_outer)
    x1_inner, y1_inner, x2_inner, y2_inner = bounding_box_inner[:4]
    x1_outer, y1_outer, x2_outer, y2_outer = bounding_box_outer[:4]
    x1 = max(x1_inner, x1_outer)
    y1 = max(y1_inner, y1_outer)
    x2 = min(x2_inner, x2_outer)
    y2 = min(y2_inner, y2_outer)
    area_intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_core = (x2_inner - x1_inner) * (y2_inner - y1_inner)
    if area_intersection > 0.9 * area_core:
        return True
    else:
        return False


def video_breakdown(VideoPath, ImgDir):
    """
    视频分解模块
    将视频按帧分解为图片保存到指定文件夹
    返回视频的帧率，宽度，高度
    :param VideoPath: 视频路径
    :param ImgDir:  图片保存路径
    :return: fps, frame_width, frame_height
    """
    cap = cv2.VideoCapture(VideoPath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    real_duration = 0.3  # 秒
    video_summarization = [[fps, total_frames, frame_width, frame_height, duration, real_duration]]
    video_summarization = pd.DataFrame(video_summarization,
                                       columns=['fps', 'total_frames', 'frame_width', 'frame_height', 'duration',
                                                'real_duration'])
    count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            img_name = os.path.join(ImgDir, '{}.jpg'.format(count))
            cv2.imwrite(img_name, frame)
            count += 1
        else:
            break
    return fps, frame_width, frame_height


def calculate_relative_displacement(now_box, before_box, Directional_significance):
    """
    计算液滴的相对位移
    :param now_box: 当前帧液滴的坐标
    :param before_box: 上一帧液滴的坐标
    :param Directional_significance: 运动方向0:向下，1:向上，2:向右，3:向左
    :return: x_displacement, y_displacement
    """
    x_displacement = 0
    y_displacement = 0
    if Directional_significance == 0:
        x_displacement = now_box[0] - before_box[0]
        y_displacement = now_box[1] - before_box[1]
        if y_displacement >= -20:
            return True
        else:
            return False
    if Directional_significance == 1:
        x_displacement = now_box[0] - before_box[0]
        y_displacement = now_box[1] - before_box[1]
        if y_displacement <= 20:
            return True
        else:
            return False
    if Directional_significance == 2:
        x_displacement = now_box[0] - before_box[0]
        y_displacement = now_box[1] - before_box[1]
        if x_displacement >= -20:
            return True
        else:
            return False
    if Directional_significance == 3:
        x_displacement = now_box[0] - before_box[0]
        y_displacement = now_box[1] - before_box[1]
        if x_displacement <= 20:
            return True
        else:
            return False


def droplet_positioning(PklResults, Directional_significance, detection_score=0.1):
    """
    液滴定位模块
    """
    id_list = []
    image_id_unique = PklResults['image_id'].unique()
    emulsion_info = []
    emulsion_boxes_list = []  # 一个总的液滴列表
    id_list_before = []  # 上一帧的液滴
    for i in tqdm(image_id_unique):
        id_list_new = []  # 需要查询的液滴
        id_list_set = set(id_list)
        id_list_before_set = set(id_list_before)
        intersection = id_list_set.intersection(id_list_before_set)
        id_list_before_set = list(id_list_before_set)
        id_list_before_set = [item for item in id_list_before_set if item not in intersection]
        id_list_set_union = id_list + id_list_before_set
        # print('id_list_before:', id_list_before)
        # print('id_list:', id_list)
        # print('id_list_before_set:', id_list_before_set)
        # print('id_list_set_union:', id_list_set_union)
        pkl_result_now = PklResults[PklResults['image_id'] == i]
        pkl_result_now = pkl_result_now.sort_values(by='bounding_box', key=lambda x: x.str[1])
        pkl_result_now = pkl_result_now.reset_index(drop=True)
        # 如果是第1帧，直接添加
        if i == 0:
            for j in range(len(pkl_result_now)):
                # 如果是单液滴，直接添加
                if pkl_result_now.loc[j, 'category_id'] == 2:
                    image_id = pkl_result_now.loc[j, 'image_id']
                    boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                    category_id = pkl_result_now.loc[j, 'category_id']
                    emulsion_info.append([[image_id, boxs, category_id]])
                    id_list_new.append(len(emulsion_info) - 1)
                    emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
                # 如果是双液滴，寻找内液滴
                if pkl_result_now.loc[j, 'category_id'] == 1:
                    # 寻找内液滴
                    boxs_core_list = []
                    for k in range(len(pkl_result_now)):
                        if pkl_result_now.loc[k, 'category_id'] == 3:
                            if match_core_shell(pkl_result_now.loc[k, 'bounding_box'],
                                                pkl_result_now.loc[j, 'bounding_box']):
                                boxs_core_list.append(pkl_result_now.loc[k, 'bounding_box'])
                    # 如果找到内液滴
                    if len(boxs_core_list) > 0:
                        image_id = pkl_result_now.loc[j, 'image_id']
                        boxs_shell = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                        boxs_core = []
                        for box in boxs_core_list:
                            boxs_core = boxs_core + box[0:4].tolist()
                        category_id = pkl_result_now.loc[j, 'category_id']
                        # 将boxs_shell和boxs_core拼接，前4个是shell，后面每4个是一个core
                        boxs = boxs_shell + boxs_core
                        emulsion_info.append([[image_id, boxs, category_id]])
                        id_list_new.append(len(emulsion_info) - 1)
                        emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
        # 如果不是第1帧.则需要判断当前帧的液滴是否在上一帧的液滴中
        else:
            for j in range(len(pkl_result_now)):
                # 如果是单液滴，与上一帧的单液滴进行匹配
                if pkl_result_now.loc[j, 'category_id'] == 2:
                    for k in id_list_set_union:
                        xy_dis = calculate_relative_displacement(pkl_result_now.loc[j, 'bounding_box'][0:4],
                                                                 emulsion_info[k][-1][1][0:4], Directional_significance)
                        if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                         emulsion_info[k][-1][1][0:4]) >= detection_score \
                                and emulsion_info[k][-1][2] == 2 and xy_dis:
                            image_id = pkl_result_now.loc[j, 'image_id']
                            boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                            category_id = pkl_result_now.loc[j, 'category_id']
                            emulsion_info[k].append([image_id, boxs, category_id])
                            emulsion_boxes_list.append([image_id, boxs, category_id, k])
                            id_list_new.append(k)
                            id_list_set_union.remove(k)
                            break
                    else:
                        image_id = pkl_result_now.loc[j, 'image_id']
                        boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                        category_id = pkl_result_now.loc[j, 'category_id']
                        # 检查是否与已存在的液滴重合
                        for k in id_list_new:
                            if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                             emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and emulsion_info[k][-1][2] == 2:
                                break
                        else:
                            emulsion_info.append([[image_id, boxs, category_id]])
                            id_list_new.append(len(emulsion_info) - 1)
                            emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
                # 如果是双液滴，与上一帧的双液滴进行匹配
                if pkl_result_now.loc[j, 'category_id'] == 1:
                    # 寻找内液滴
                    boxs_core_list = []
                    for k in range(len(pkl_result_now)):
                        if pkl_result_now.loc[k, 'category_id'] == 3:
                            if match_core_shell(pkl_result_now.loc[k, 'bounding_box'],
                                                pkl_result_now.loc[j, 'bounding_box']):
                                boxs_core_list.append(pkl_result_now.loc[k, 'bounding_box'])
                    # 如果找到内液滴
                    if len(boxs_core_list) > 0:
                        for k in id_list_set_union:
                            xy_dis = calculate_relative_displacement(pkl_result_now.loc[j, 'bounding_box'][0:4],
                                                                     emulsion_info[k][-1][1][0:4],
                                                                     Directional_significance)
                            if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                             emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and pkl_result_now.loc[j, 'category_id'] == emulsion_info[k][-1][2] \
                                    and xy_dis:
                                image_id = pkl_result_now.loc[j, 'image_id']
                                boxs_shell = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                                boxs_core = []
                                for box in boxs_core_list:
                                    boxs_core = boxs_core + box[0:4].tolist()
                                category_id = pkl_result_now.loc[j, 'category_id']
                                # 将boxs_shell和boxs_core拼接，前4个是shell，后面每4个是一个core
                                boxs = boxs_shell + boxs_core
                                emulsion_info[k].append([image_id, boxs, category_id])
                                emulsion_boxes_list.append([image_id, boxs, category_id, k])
                                id_list_new.append(k)
                                id_list_set_union.remove(k)
                                break
                            elif calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                               emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and emulsion_info[k][-1][2] == 2 and xy_dis:
                                image_id = pkl_result_now.loc[j, 'image_id']
                                boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                                category_id = 2
                                emulsion_info[k].append([image_id, boxs, category_id])
                                emulsion_boxes_list.append([image_id, boxs, category_id, k])
                                id_list_new.append(k)
                                id_list_set_union.remove(k)
                                break


                        else:
                            image_id = pkl_result_now.loc[j, 'image_id']
                            boxs_shell = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                            boxs_core = []
                            for box in boxs_core_list:
                                boxs_core = boxs_core + box[0:4].tolist()
                            category_id = pkl_result_now.loc[j, 'category_id']
                            # 将boxs_shell和boxs_core拼接，前4个是shell，后面每4个是一个core
                            boxs = boxs_shell + boxs_core
                            emulsion_info.append([[image_id, boxs, category_id]])
                            id_list_new.append(len(emulsion_info) - 1)
                            emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
                    else:
                        for k in id_list_set_union:
                            if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                             emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and emulsion_info[k][-1][2] == 2:
                                image_id = pkl_result_now.loc[j, 'image_id']
                                boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                                category_id = 2
                                emulsion_info[k].append([image_id, boxs, category_id])
                                emulsion_boxes_list.append([image_id, boxs, category_id, k])
                                id_list_new.append(k)
                                id_list_set_union.remove(k)
                                break
        id_list_before = id_list
        id_list = id_list_new
    emulsion_boxes_list = pd.DataFrame(emulsion_boxes_list, columns=['image_id', 'boxs', 'category_id', 'emulsion_id'])
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='image_id')
    emulsion_boxes_list = emulsion_boxes_list.reset_index(drop=True)
    return emulsion_info, emulsion_boxes_list


def calculate_directional(PklResults, detection_score=0.1):
    """
    计算液滴的运动方向
    :param PklResults:
    :return: 0:向下，1:向上，2:向右，3:向左
    """
    image_id_unique = PklResults['image_id'].unique()
    emulsion_info = []
    emulsion_boxes_list = []  # 一个总的液滴列表
    for i in tqdm(image_id_unique):
        id_list_new = []  # 需要查询的液滴
        pkl_result_now = PklResults[PklResults['image_id'] == i]
        pkl_result_now = pkl_result_now.sort_values(by='bounding_box', key=lambda x: x.str[1])
        pkl_result_now = pkl_result_now.reset_index(drop=True)
        # 如果是第1帧，直接添加
        if i == 0:
            for j in range(len(pkl_result_now)):
                # 如果是单液滴，直接添加
                if pkl_result_now.loc[j, 'category_id'] == 2:
                    image_id = pkl_result_now.loc[j, 'image_id']
                    boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                    category_id = pkl_result_now.loc[j, 'category_id']
                    emulsion_info.append([[image_id, boxs, category_id]])
                    id_list_new.append(len(emulsion_info) - 1)
                    emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
                # 如果是双液滴，寻找内液滴
                if pkl_result_now.loc[j, 'category_id'] == 1:
                    # 寻找内液滴
                    boxs_core_list = []
                    for k in range(len(pkl_result_now)):
                        if pkl_result_now.loc[k, 'category_id'] == 3:
                            if match_core_shell(pkl_result_now.loc[k, 'bounding_box'],
                                                pkl_result_now.loc[j, 'bounding_box']):
                                boxs_core_list.append(pkl_result_now.loc[k, 'bounding_box'])
                    # 如果找到内液滴
                    if len(boxs_core_list) > 0:
                        image_id = pkl_result_now.loc[j, 'image_id']
                        boxs_shell = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                        boxs_core = []
                        for box in boxs_core_list:
                            boxs_core = boxs_core + box[0:4].tolist()
                        category_id = pkl_result_now.loc[j, 'category_id']
                        # 将boxs_shell和boxs_core拼接，前4个是shell，后面每4个是一个core
                        boxs = boxs_shell + boxs_core
                        emulsion_info.append([[image_id, boxs, category_id]])
                        id_list_new.append(len(emulsion_info) - 1)
                        emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
        # 如果不是第1帧.则需要判断当前帧的液滴是否在上一帧的液滴中
        else:
            for j in range(len(pkl_result_now)):
                # 如果是单液滴，与上一帧的单液滴进行匹配
                if pkl_result_now.loc[j, 'category_id'] == 2:
                    for k in id_list:
                        if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                         emulsion_info[k][-1][1][0:4]) >= detection_score \
                                and emulsion_info[k][-1][2] == 2:
                            image_id = pkl_result_now.loc[j, 'image_id']
                            boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                            category_id = pkl_result_now.loc[j, 'category_id']
                            emulsion_info[k].append([image_id, boxs, category_id])
                            emulsion_boxes_list.append([image_id, boxs, category_id, k])
                            id_list_new.append(k)
                            id_list.remove(k)
                            break
                    else:
                        image_id = pkl_result_now.loc[j, 'image_id']
                        boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                        category_id = pkl_result_now.loc[j, 'category_id']
                        # 检查是否与已存在的液滴重合
                        for k in id_list_new:
                            if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                             emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and emulsion_info[k][-1][2] == 2:
                                break
                        else:
                            emulsion_info.append([[image_id, boxs, category_id]])
                            id_list_new.append(len(emulsion_info) - 1)
                            emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
                # 如果是双液滴，与上一帧的双液滴进行匹配
                if pkl_result_now.loc[j, 'category_id'] == 1:
                    # 寻找内液滴
                    boxs_core_list = []
                    for k in range(len(pkl_result_now)):
                        if pkl_result_now.loc[k, 'category_id'] == 3:
                            if match_core_shell(pkl_result_now.loc[k, 'bounding_box'],
                                                pkl_result_now.loc[j, 'bounding_box']):
                                boxs_core_list.append(pkl_result_now.loc[k, 'bounding_box'])
                    # 如果找到内液滴
                    if len(boxs_core_list) > 0:
                        for k in id_list:
                            if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                             emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and pkl_result_now.loc[j, 'category_id'] == emulsion_info[k][-1][2]:
                                image_id = pkl_result_now.loc[j, 'image_id']
                                boxs_shell = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                                boxs_core = []
                                for box in boxs_core_list:
                                    boxs_core = boxs_core + box[0:4].tolist()
                                category_id = pkl_result_now.loc[j, 'category_id']
                                # 将boxs_shell和boxs_core拼接，前4个是shell，后面每4个是一个core
                                boxs = boxs_shell + boxs_core
                                emulsion_info[k].append([image_id, boxs, category_id])
                                emulsion_boxes_list.append([image_id, boxs, category_id, k])
                                id_list_new.append(k)
                                id_list.remove(k)
                                break
                            elif calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                               emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and emulsion_info[k][-1][2] == 2:
                                image_id = pkl_result_now.loc[j, 'image_id']
                                boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                                category_id = 2
                                emulsion_info[k].append([image_id, boxs, category_id])
                                emulsion_boxes_list.append([image_id, boxs, category_id, k])
                                id_list_new.append(k)
                                id_list.remove(k)
                                break


                        else:
                            image_id = pkl_result_now.loc[j, 'image_id']
                            boxs_shell = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                            boxs_core = []
                            for box in boxs_core_list:
                                boxs_core = boxs_core + box[0:4].tolist()
                            category_id = pkl_result_now.loc[j, 'category_id']
                            # 将boxs_shell和boxs_core拼接，前4个是shell，后面每4个是一个core
                            boxs = boxs_shell + boxs_core
                            emulsion_info.append([[image_id, boxs, category_id]])
                            id_list_new.append(len(emulsion_info) - 1)
                            emulsion_boxes_list.append([image_id, boxs, category_id, len(emulsion_info) - 1])
                    else:
                        for k in id_list:
                            if calculate_iou(pkl_result_now.loc[j, 'bounding_box'],
                                             emulsion_info[k][-1][1][0:4]) >= detection_score \
                                    and emulsion_info[k][-1][2] == 2:
                                image_id = pkl_result_now.loc[j, 'image_id']
                                boxs = pkl_result_now.loc[j, 'bounding_box'][0:4].tolist()
                                category_id = 2
                                emulsion_info[k].append([image_id, boxs, category_id])
                                emulsion_boxes_list.append([image_id, boxs, category_id, k])
                                id_list_new.append(k)
                                id_list.remove(k)
                                break

        id_list = id_list_new
    emulsion_boxes_list = pd.DataFrame(emulsion_boxes_list, columns=['image_id', 'boxs', 'category_id', 'emulsion_id'])
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='image_id')
    emulsion_boxes_list = emulsion_boxes_list.reset_index(drop=True)
    # 计算每个液滴第1帧的和最后一帧的x1, x2, y1, y2
    emulsion_info_x1_x2_y1_y2 = []
    for i in range(len(emulsion_info)):
        boxes_sum = emulsion_info[i]
        boxes = boxes_sum[0][1]
        x1 = boxes[0]
        x2 = boxes[2]
        y1 = boxes[1]
        y2 = boxes[3]
        boxes = boxes_sum[-1][1]
        x1_end = boxes[0]
        x2_end = boxes[2]
        y1_end = boxes[1]
        y2_end = boxes[3]
        emulsion_info_x1_x2_y1_y2.append([x1, x2, y1, y2, x1_end, x2_end, y1_end, y2_end])
    emulsion_info_x1_x2_y1_y2 = pd.DataFrame(emulsion_info_x1_x2_y1_y2,
                                             columns=['x1', 'x2', 'y1', 'y2', 'x1_end', 'x2_end', 'y1_end', 'y2_end'])
    # 计算第1帧和最后一帧的液滴的x1, x2, y1, y2的差值
    emulsion_info_x1_x2_y1_y2['x1_diff'] = emulsion_info_x1_x2_y1_y2['x1_end'] - emulsion_info_x1_x2_y1_y2['x1']
    emulsion_info_x1_x2_y1_y2['x2_diff'] = emulsion_info_x1_x2_y1_y2['x2_end'] - emulsion_info_x1_x2_y1_y2['x2']
    emulsion_info_x1_x2_y1_y2['y1_diff'] = emulsion_info_x1_x2_y1_y2['y1_end'] - emulsion_info_x1_x2_y1_y2['y1']
    emulsion_info_x1_x2_y1_y2['y2_diff'] = emulsion_info_x1_x2_y1_y2['y2_end'] - emulsion_info_x1_x2_y1_y2['y2']  #
    # 计算x1_diff, x2_diff, y1_diff, y2_diff的平均值
    x1_diff_mean = emulsion_info_x1_x2_y1_y2['x1_diff'].mean()
    x2_diff_mean = emulsion_info_x1_x2_y1_y2['x2_diff'].mean()
    y1_diff_mean = emulsion_info_x1_x2_y1_y2['y1_diff'].mean()
    y2_diff_mean = emulsion_info_x1_x2_y1_y2['y2_diff'].mean()
    Directional_significance = 0  # 默认是0，表示液滴是从上往下移动
    # 根据平均值判断液滴的运动方向，# 首先判断是x方向还是y方向
    if y1_diff_mean * y2_diff_mean > 0:
        if x1_diff_mean * x2_diff_mean < 0:
            if y1_diff_mean > 0:
                print('液滴运动方向为向下')
                Directional_significance = 0
            else:
                print('液滴运动方向为向上')
                Directional_significance = 1
        elif x1_diff_mean * x2_diff_mean > 0:
            # 如果y1_diff_mean和y2_diff_mean的绝对值的和大于x1_diff_mean和x2_diff_mean的绝对值的和的5倍，则认为液滴运动方向为y方向
            if abs(y1_diff_mean) + abs(y2_diff_mean) > (abs(x1_diff_mean) + abs(x2_diff_mean)):
                if y1_diff_mean > 0:
                    print('液滴运动方向为向下')
                    Directional_significance = 0
                else:
                    print('液滴运动方向为向上')
                    Directional_significance = 1
            else:
                if x1_diff_mean > 0:
                    print('液滴运动方向为向右')
                    Directional_significance = 2
                else:
                    print('液滴运动方向为向左')
                    Directional_significance = 3
    else:
        if x1_diff_mean * x2_diff_mean > 0:
            print('液滴运动方向为x方向')
            if x1_diff_mean > 0:
                print('液滴运动方向为向右')
                Directional_significance = 2
            else:
                print('液滴运动方向为向左')
                Directional_significance = 3
        elif x1_diff_mean * x2_diff_mean < 0:
            # 如果x1_diff_mean和x2_diff_mean的绝对值的和大于y1_diff_mean和y2_diff_mean的绝对值的和的5倍，则认为液滴运动方向为x方向
            if abs(x1_diff_mean) + abs(x2_diff_mean) > (abs(y1_diff_mean) + abs(y2_diff_mean)):
                if x1_diff_mean > 0:
                    print('液滴运动方向为向右')
                    Directional_significance = 2
                else:
                    print('液滴运动方向为向左')
                    Directional_significance = 3
            else:
                if y1_diff_mean > 0:
                    print('液滴运动方向为向下')
                    Directional_significance = 0
                else:
                    print('液滴运动方向为向上')
                    Directional_significance = 1
    return Directional_significance


def image_statistical(img_path, pkl_path, out_path, statistical_mode='Double emulsion',algorithm=None):
    """
    输出检测结果
    :param statistical_mode:
    :param img_path: 图片路径
    :param pkl_path: 检测结果路径
    :param out_path: 输出图片路径
    :return:
    """

    classname_to_id = {
        "double-o": 1,
        'single': 2,
        'double-i': 3, }
    id_to_classname = {v: k for k, v in classname_to_id.items()}
    data_summary = pd.DataFrame(columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    try:
        assert os.path.exists(img_path)
        assert os.path.exists(pkl_path)
    except:
        print('please check the path')
        return
    pkl_result = read_pkl(pkl_path, nms_flag=True, short_long_ratio_flag=False, short_long_ratio_threshold=0.8)
    pkl_result_pd = pd.DataFrame(pkl_result, columns=['image_id', 'category_id', 'bounding_box'])
    color_map = {1: (220, 20, 60), 2: (0, 0, 235), 3: (0, 0, 0)}
    img = cv2.imread(img_path)
    for i in range(len(pkl_result_pd)):
        category_id = pkl_result_pd.loc[i, 'category_id']
        bounding_box = pkl_result_pd.loc[i, 'bounding_box']
        x1, y1, x2, y2 = bounding_box[:4]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color_map[category_id], 2)
    cv2.imwrite(out_path, img)
    if statistical_mode == 'Single droplet':
        # 只统计单乳液，并且只统计直径
        diameter_list = []
        for i in range(len(pkl_result_pd)):
            category_id = pkl_result_pd.loc[i, 'category_id']
            if category_id == 2:
                bounding_box = pkl_result_pd.loc[i, 'bounding_box']
                diameter = calculate_diameter(bounding_box)
                diameter_list.append(diameter)
        if len(diameter_list) == 0:
            print('There is no single droplet in the image')
            data_summary = None
            return data_summary
        else:
            diameter_pd = pd.DataFrame(diameter_list, columns=['single_diameter'])
            diameter_describe = data_describe(diameter_pd, 'single_diameter')
        # 将结果保存到data_summary
        data_summary = pd.concat([data_summary, diameter_describe], axis=0)
        return data_summary
    elif statistical_mode == 'Double emulsion' or statistical_mode == 'Cell encapsulation' or statistical_mode == 'Single-cell encapsulation':
        pkl_result = read_pkl(pkl_path, nms_flag=True, short_long_ratio_flag=True, short_long_ratio_threshold=0.6)
        pkl_result = pd.DataFrame(pkl_result, columns=['image_id', 'category_id', 'bounding_box'])
        double_info = []
        single_info = []
        core_num_list = []
        for i in range(len(pkl_result)):
            if pkl_result.loc[i, 'category_id'] == 1:
                core_boxes_list = []
                core_diameter_list = []
                for j in range(len(pkl_result)):
                    if pkl_result.loc[j, 'category_id'] == 3:
                        if match_core_shell(pkl_result.loc[j, 'bounding_box'], pkl_result.loc[i, 'bounding_box']):
                            core_boxes_list.append(pkl_result.loc[j, 'bounding_box'])
                            core_diameter_list.append(calculate_diameter(pkl_result.loc[j, 'bounding_box']))
                core_num = len(core_boxes_list)
                if core_num == 0:
                    continue
                core_num_list.append(core_num)
                if core_num == 1:
                    core_diameter = core_diameter_list
                    shell_diameter = calculate_diameter(pkl_result.loc[i, 'bounding_box'])
                    core_box = core_boxes_list[0]
                    concentricity = calculate_concentricity(pkl_result.loc[i, 'bounding_box'], core_box)  # 同心度
                    core_to_shell_ratio = [
                        calculate_core_to_shell_ratio(core_box, pkl_result.loc[i, 'bounding_box'])]  # 核壳比
                    volume_ratio = calculate_volume_ratio(core_boxes_list, pkl_result.loc[i, 'bounding_box'])
                    double_info.append(
                        [pkl_result.loc[i, 'bounding_box'], core_box, core_diameter, shell_diameter, concentricity,
                         core_to_shell_ratio, volume_ratio])
                elif core_num > 1:
                    # 这一部分还没有写完！！
                    core_diameter = core_diameter_list
                    shell_diameter = calculate_diameter(pkl_result.loc[i, 'bounding_box'])
                    # 有多个核，不计算同心度，不计算核壳比，计算体积比
                    concentricity = None
                    core_to_shell_ratio = None
                    volume_ratio = calculate_volume_ratio(core_boxes_list, pkl_result.loc[i, 'bounding_box'])
                    double_info.append(
                        [pkl_result.loc[i, 'bounding_box'], core_boxes_list, core_diameter, shell_diameter,
                         concentricity,
                         core_to_shell_ratio, volume_ratio])

                else:
                    continue
            if pkl_result.loc[i, 'category_id'] == 2:
                single_diameter = calculate_diameter(pkl_result.loc[i, 'bounding_box'])
                single_info.append([pkl_result.loc[i, 'bounding_box'], single_diameter])
        # 转换为DataFrame
        double_info = pd.DataFrame(double_info, columns=['double_box', 'core_box', 'core_diameter', 'shell_diameter',
                                                         'concentricity', 'core_to_shell_ratio', 'volume_ratio'])
        single_info = pd.DataFrame(single_info, columns=['single_box', 'single_diameter'])

        # 计算double_box的数量
        double_box_num = len(double_info)
        # 计算核的数量
        core_num = 0
        for i in range(double_box_num):
            core_num += len(double_info.loc[i, 'core_diameter'])
        # 计算核的直径信息
        core_diameter_list = []
        for i in range(double_box_num):
            for j in range(len(double_info.loc[i, 'core_diameter'])):
                # 如果不为空
                if double_info.loc[i, 'core_diameter'][j] is not None:
                    core_diameter_list.append(double_info.loc[i, 'core_diameter'][j])
                else:
                    print('warning: core_diameter is None')
        core_diameter_pd = pd.DataFrame(core_diameter_list, columns=['core_diameter'])
        # 如果不为空
        if len(core_diameter_pd) > 0:
            core_diameter_describe = data_describe(core_diameter_pd, 'core_diameter')
        else:
            core_diameter_describe = pd.DataFrame(columns=['Variable name', 'Count', 'Average value',
                                                           'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
        # 计算核的数量信息
        core_num_pd = pd.DataFrame(core_num_list, columns=['core_num'])
        # 统计核数量为1的数量
        core_num_single = []
        core_num_multi = []
        for i in range(len(core_num_list)):
            if core_num_list[i] == 1:
                core_num_single.append(core_num_list[i])
            else:
                core_num_multi.append(core_num_list[i])
        core_num_single = pd.DataFrame(core_num_single, columns=['core_num_single'])
        core_num_multi = pd.DataFrame(core_num_multi, columns=['core_num_mutli'])
        # 如果不为空
        if len(core_num_single) > 0:
            core_num_single_describe = data_describe(core_num_single, 'core_num_single')
        else:
            core_num_single_describe = pd.DataFrame(columns=['Variable name', 'Count', 'Average value',
                                                             'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
        if len(core_num_multi) > 0:
            core_num_multi_describe = data_describe(core_num_multi, 'core_num_mutli')
        else:
            core_num_multi_describe = pd.DataFrame(columns=['Variable name', 'Count', 'Average value',
                                                            'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
        if len(core_num_pd) > 0:
            core_num_describe = data_describe(core_num_pd, 'core_num')
        else:
            core_num_describe = pd.DataFrame(columns=['Variable name', 'Count', 'Average value',
                                                      'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
        # 计算壳的直径信息
        shell_diameter_pd = double_info['shell_diameter']
        shell_diameter_pd = pd.DataFrame(shell_diameter_pd, columns=['shell_diameter'])
        if len(shell_diameter_pd) > 0:
            shell_diameter_describe = data_describe(shell_diameter_pd, 'shell_diameter')
        else:
            shell_diameter_describe = pd.DataFrame(columns=['Variable name', 'Count', 'Average value',
                                                           'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
        # 计算同心度,如果核的数量为1，则计算同心度和核壳比
        concentricity_list = []
        core_to_shell_ratio_list = []
        volume_ratio_list = []
        for i in range(double_box_num):
            if len(double_info.loc[i, 'core_diameter']) == 1:
                concentricity_list.append(double_info.loc[i, 'concentricity'])
                core_to_shell_ratio_list.append(double_info.loc[i, 'core_to_shell_ratio'])
            elif len(double_info.loc[i, 'core_diameter']) > 1:
                volume_ratio_list.append(double_info.loc[i, 'volume_ratio'])
        # 如果不为空
        if len(concentricity_list) > 0:
            concentricity_pd = pd.DataFrame(double_info['concentricity'], columns=['concentricity'])
            # 删除空值
            concentricity_pd = concentricity_pd.dropna()
            concentricity_describe = data_describe(concentricity_pd, 'concentricity')
        else:
            concentricity_describe = []
        if len(core_to_shell_ratio_list) > 0:
            core_to_shell_ratio_list_pd = pd.DataFrame(core_to_shell_ratio_list, columns=['core_to_shell_ratio'])
            # 删除空值
            core_to_shell_ratio_list_pd = core_to_shell_ratio_list_pd.dropna()
            core_to_shell_ratio_describe = data_describe(core_to_shell_ratio_list_pd, 'core_to_shell_ratio')
        else:
            core_to_shell_ratio_describe = []
        if len(volume_ratio_list) > 0:
            volume_ratio_pd = pd.DataFrame(volume_ratio_list, columns=['volume_ratio'])
            # 删除空值
            volume_ratio_pd = volume_ratio_pd.dropna()
            volume_ratio_describe = data_describe(volume_ratio_pd, 'volume_ratio')
        else:
            volume_ratio_describe = []
        # 计算单乳液信息，如果有的话
        if len(single_info) > 0:
            single_diameter_pd = single_info['single_diameter']
            single_diameter_pd = pd.DataFrame(single_diameter_pd, columns=['single_diameter'])
            single_diameter_describe = data_describe(single_diameter_pd, 'single_diameter')
        else:
            single_diameter_describe = []
        summary_list = [single_diameter_describe, core_diameter_describe, shell_diameter_describe,
                        concentricity_describe,
                        core_to_shell_ratio_describe, volume_ratio_describe, core_num_describe, core_num_single_describe,
                        core_num_multi_describe]
        print(data_summary)
        for i in range(len(summary_list)):
            if len(summary_list[i]) != 0:
                data_summary = pd.concat([data_summary, summary_list[i]], axis=0)

        print('data_summary:', data_summary)
    else:
        print('statistical_mode is wrong')
        print('statistical_mode:', statistical_mode)
        data_summary = pd.DataFrame(columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                             'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    return data_summary



def video_statistical(video_path, pkl_path, video_out_path, statistical_mode='Double emulsion', time_span=1,
                      flow_direction='top-to-bottom',
                      reference_line=30):
    """
    输出检测结果
    :param reference_line:
    :param flow_direction:
    :param time_span:
    :param statistical_mode:
    :param video_path: 视频路径
    :param pkl_path: 检测结果路径
    :param video_out_path: 输出图片路径
    :return:
    """
    classname_to_id = {
        "double-o": 1,
        'single': 2,
        'double-i': 3, }
    flow_direction_to_id = {
        'top-to-bottom': 0,
        'bottom-to-top': 1,
        'left-to-right': 2,
        'right-to-left': 3,
    }
    # 检查video_out_path后缀是否为.mp4，如果不是，替换为.mp4
    if video_out_path.split('.')[-1] != 'mp4':
        video_out_path = video_out_path.split('.')[0] + '.mp4'
    id_to_classname = {v: k for k, v in classname_to_id.items()}
    data_summary = pd.DataFrame()
    # SystemExit path check
    try:
        assert os.path.exists(video_path)
        assert os.path.exists(pkl_path)
    except:
        print('please check the path')
        return
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    pkl_result = read_pkl(pkl_path, nms_flag=True, short_long_ratio_flag=True, short_long_ratio_threshold=0.6)
    pkl_result = pd.DataFrame(pkl_result, columns=['image_id', 'category_id', 'bounding_box'])
    Directional_significance = flow_direction_to_id[flow_direction]
    emulsion_info, emulsion_boxes_list = droplet_positioning(pkl_result, Directional_significance)
    image_id_unique = emulsion_boxes_list['image_id'].unique()
    Judgment_line = reference_line
    y1_list = []
    y2_list = []
    x1_list = []
    x2_list = []
    for i in range(len(emulsion_info)):
        boxes = emulsion_info[i][-1]
        x1 = boxes[1][0]
        x2 = boxes[1][2]
        y1 = boxes[1][1]
        y2 = boxes[1][3]
        x1_list.append(x1)
        x2_list.append(x2)
        y1_list.append(y1)
        y2_list.append(y2)
    x1_list = pd.DataFrame(x1_list, columns=['x1'])
    x2_list = pd.DataFrame(x2_list, columns=['x2'])
    y1_list = pd.DataFrame(y1_list, columns=['y1'])
    y2_list = pd.DataFrame(y2_list, columns=['y2'])
    # 使用平均值作为判断线
    #  emulsion_boxes_list 添加新的一列，为y1.
    emulsion_boxes_list['y1'] = emulsion_boxes_list['boxs'].apply(lambda x: x[1])
    emulsion_boxes_list['y2'] = emulsion_boxes_list['boxs'].apply(lambda x: x[3])
    emulsion_boxes_list['x1'] = emulsion_boxes_list['boxs'].apply(lambda x: x[0])
    emulsion_boxes_list['x2'] = emulsion_boxes_list['boxs'].apply(lambda x: x[2])
    emulsion_boxes_list['x2-x1'] = emulsion_boxes_list['x2'] - emulsion_boxes_list['x1']
    emulsion_boxes_list['y2-y1'] = emulsion_boxes_list['y2'] - emulsion_boxes_list['y1']
    Judgment_line_x1 = x1_list.mean()
    Judgment_line_x2 = x2_list.mean()
    Judgment_line_y1 = y1_list.mean()
    Judgment_line_y2 = y2_list.mean()
    Judgment_line_x1 = float(Judgment_line_x1)
    Judgment_line_x2 = float(Judgment_line_x2)
    Judgment_line_y1 = float(Judgment_line_y1)
    Judgment_line_y2 = float(Judgment_line_y2)
    # 获取视频信息
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if Directional_significance == 0 or Directional_significance == 1:
        Judgment_line = Judgment_line * video_height / 100
    elif Directional_significance == 2 or Directional_significance == 3:
        Judgment_line = Judgment_line * video_width / 100
    else:
        print('Directional_significance is wrong')
    if Directional_significance == 0:
        # 计算每个液滴的y1和judgment_line的差值
        emulsion_boxes_list['y1_judge'] = emulsion_boxes_list['y1'] - Judgment_line
        # 计算每个液滴的y2和judgment_line的差值
        emulsion_boxes_list['y2_judge'] = emulsion_boxes_list['y2'] - Judgment_line
        emulsion_boxes_list['y2-y1'] = emulsion_boxes_list['y2'] - emulsion_boxes_list['y1']
        # 新加一列，计算对应emulsion_id y2-y1的平均值
        emulsion_boxes_list['avg_y2-y1'] = emulsion_boxes_list['emulsion_id'].map(
            emulsion_boxes_list.groupby('emulsion_id')['y2-y1'].mean())
        # 计算超出的比例
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['y2_judge'] / emulsion_boxes_list['avg_y2-y1']
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['y2_judge'] / emulsion_boxes_list['y2-y1']
        # 超出的比例如果是负数，说明在判断线上方，记为0，如果是正数，说明在判断线下方，最大值为1
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 0 if x < 0 else x)
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 1 if x > 1 else x)
    if Directional_significance == 1:
        # 计算每个液滴的y1和judgment_line的差值
        emulsion_boxes_list['y1_judge'] = emulsion_boxes_list['y1'] - Judgment_line
        # 计算每个液滴的y2和judgment_line的差值
        emulsion_boxes_list['y2_judge'] = emulsion_boxes_list['y2'] - Judgment_line
        emulsion_boxes_list['y2-y1'] = emulsion_boxes_list['y2'] - emulsion_boxes_list['y1']
        # 新加一列，计算对应emulsion_id y2-y1的平均值
        emulsion_boxes_list['avg_y2-y1'] = emulsion_boxes_list['emulsion_id'].map(
            emulsion_boxes_list.groupby('emulsion_id')['y2-y1'].mean())
        # 计算超出的比例
        emulsion_boxes_list['ratio'] = - (emulsion_boxes_list['y1_judge'] / emulsion_boxes_list['avg_y2-y1'])
        emulsion_boxes_list['ratio'] = - (emulsion_boxes_list['y1_judge'] / emulsion_boxes_list['y2-y1'])
        # 超出的比例如果是负数，说明在判断线上方，记为0，如果是正数，说明在判断线下方，最大值为1
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 0 if x < 0 else x)
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 1 if x > 1 else x)
    if Directional_significance == 2:
        # 计算每个液滴的x1和judgment_line的差值
        emulsion_boxes_list['x1_judge'] = emulsion_boxes_list['x1'] - Judgment_line
        # 计算每个液滴的x2和judgment_line的差值
        emulsion_boxes_list['x2_judge'] = emulsion_boxes_list['x2'] - Judgment_line
        emulsion_boxes_list['x2-x1'] = emulsion_boxes_list['x2'] - emulsion_boxes_list['x1']
        # 新加一列，计算对应emulsion_id x2-x1的平均值
        emulsion_boxes_list['avg_x2-x1'] = emulsion_boxes_list['emulsion_id'].map(
            emulsion_boxes_list.groupby('emulsion_id')['x2-x1'].mean())
        # 计算超出的比例
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['x2_judge'] / emulsion_boxes_list['avg_x2-x1']
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['x2_judge'] / emulsion_boxes_list['x2-x1']
        # 超出的比例如果是负数，说明在判断线左方，记为0，如果是正数，说明在判断线右方，最大值为1
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 0 if x < 0 else x)
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 1 if x > 1 else x)
    if Directional_significance == 3:
        # 计算每个液滴的x1和judgment_line的差值
        emulsion_boxes_list['x1_judge'] = emulsion_boxes_list['x1'] - Judgment_line
        # 计算每个液滴的x2和judgment_line的差值
        emulsion_boxes_list['x2_judge'] = emulsion_boxes_list['x2'] - Judgment_line
        emulsion_boxes_list['x2-x1'] = emulsion_boxes_list['x2'] - emulsion_boxes_list['x1']
        # 新加一列，计算对应emulsion_id x2-x1的平均值
        emulsion_boxes_list['avg_x2-x1'] = emulsion_boxes_list['emulsion_id'].map(
            emulsion_boxes_list.groupby('emulsion_id')['x2-x1'].mean())
        # 计算超出的比例
        emulsion_boxes_list['ratio'] = - (emulsion_boxes_list['x1_judge'] / emulsion_boxes_list['avg_x2-x1'])
        emulsion_boxes_list['ratio'] = - (emulsion_boxes_list['x1_judge'] / emulsion_boxes_list['x2-x1'])
        # 超出的比例如果是负数，说明在判断线左方，记为0，如果是正数，说明在判断线右方，最大值为1
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 0 if x < 0 else x)
        emulsion_boxes_list['ratio'] = emulsion_boxes_list['ratio'].apply(lambda x: 1 if x > 1 else x)
    # 按image_id从小到大排序
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='image_id')
    # 按emulsion_id从小到大排序
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='emulsion_id')
    # 重置索引
    emulsion_boxes_list = emulsion_boxes_list.reset_index(drop=True)
    emulsion_boxes_list_copy = emulsion_boxes_list.copy()
    # 为双液滴计算内液滴的数量
    emulsion_boxes_list_copy['core_num'] = 0
    for i in range(len(emulsion_boxes_list_copy)):
        if emulsion_boxes_list_copy.loc[i, 'category_id'] == 1:
            core_num = (len(emulsion_boxes_list_copy.loc[i, 'boxs']) - 4) / 4
            emulsion_boxes_list_copy.loc[i, 'core_num'] = core_num
    # 对于同一个id的双液滴，如果内液滴数量有变化，只保留数量最多的那组，如果数量一样多，则所有都保留
    emulsion_boxes_list_copy['core_num'] = emulsion_boxes_list_copy['core_num'].astype(int)
    max_core_num = []
    # 按emulsion_id分组
    for i in emulsion_boxes_list_copy['emulsion_id'].unique():
        emulsion_boxes_list_copy_group = emulsion_boxes_list_copy[emulsion_boxes_list_copy['emulsion_id'] == i]
        emulsion_boxes_list_copy_group = emulsion_boxes_list_copy_group.reset_index(drop=True)
        core_num_list = emulsion_boxes_list_copy_group['core_num'].unique()
        if len(core_num_list) > 1:
            core_num_max = max(core_num_list)
            # 记住emlusion_id 和 core_num_max
            max_core_num.append([i, core_num_max])
        else:
            max_core_num.append([i, core_num_list[0]])
    emulsion_boxes_list_copy = emulsion_boxes_list_copy.reset_index(drop=True)
    # 删除core_num不是最大值的行
    for i in range(len(max_core_num)):
        emulsion_id = max_core_num[i][0]
        core_num = max_core_num[i][1]
        emulsion_boxes_list_copy_group = emulsion_boxes_list_copy[
            emulsion_boxes_list_copy['emulsion_id'] == emulsion_id]
        for j in emulsion_boxes_list_copy_group.index:
            if emulsion_boxes_list_copy_group.loc[j, 'core_num'] != core_num:
                emulsion_boxes_list_copy = emulsion_boxes_list_copy.drop(index=j)
    emulsion_boxes_list_copy = emulsion_boxes_list_copy.reset_index(drop=True)
    # 计算
    emulsion_boxes_list_copy['diameter'] = None  # 单液滴的直径
    emulsion_boxes_list_copy['concentricity'] = None  # 双液滴的同心度
    emulsion_boxes_list_copy['core_shell_ratio'] = None  # 双液滴的核壳比
    emulsion_boxes_list_copy['volume_ratio'] = None  # 双液滴的体积比
    core_diameter = []  # 所有内液滴的直径
    for i in range(len(emulsion_boxes_list_copy)):
        # 如果是单液滴
        if emulsion_boxes_list_copy.loc[i, 'category_id'] == 2:
            # 计算液滴直径
            x1, y1, x2, y2 = emulsion_boxes_list_copy.loc[i, 'boxs'][0:4]
            diameter = calculate_diameter([x1, y1, x2, y2])
            emulsion_boxes_list_copy.loc[i, 'diameter'] = diameter
        # 如果是双液滴
        if emulsion_boxes_list_copy.loc[i, 'category_id'] == 1:
            # 计算液滴外径
            x1, y1, x2, y2 = emulsion_boxes_list_copy.loc[i, 'boxs'][0:4]
            diameter = calculate_diameter([x1, y1, x2, y2])
            emulsion_boxes_list_copy.loc[i, 'diameter'] = diameter
            # 计算内液滴数量
            core_num = (len(emulsion_boxes_list_copy.loc[i, 'boxs']) - 4) / 4
            emulsion_boxes_list_copy.loc[i, 'core_num'] = core_num
            # 计算内液滴直径
            for j in range(4, len(emulsion_boxes_list_copy.loc[i, 'boxs']), 4):
                x1, y1, x2, y2 = emulsion_boxes_list_copy.loc[i, 'boxs'][j:j + 4]
                diameter = calculate_diameter([x1, y1, x2, y2])
                core_diameter.append(diameter)
            # 如果内液滴数量等于1，计算同心度和核壳比
            if core_num == 1:
                core_box = emulsion_boxes_list_copy.loc[i, 'boxs'][4:8]
                shell_box = emulsion_boxes_list_copy.loc[i, 'boxs'][0:4]
                concentricity = calculate_concentricity(core_box, shell_box)
                emulsion_boxes_list_copy.loc[i, 'concentricity'] = concentricity
                # 计算核壳比
                core_shell_ratio = calculate_core_to_shell_ratio(core_box, shell_box)
                emulsion_boxes_list_copy.loc[i, 'core_shell_ratio'] = core_shell_ratio
            # 如果内液滴数量大于1，计算体积比
            if core_num > 1:
                # 先获取内液滴的box
                core_box = []
                shell_box = emulsion_boxes_list_copy.loc[i, 'boxs'][0:4]
                for j in range(4, len(emulsion_boxes_list_copy.loc[i, 'boxs']), 4):
                    core_box.append(emulsion_boxes_list_copy.loc[i,]['boxs'][j:j + 4])
                # 计算体积比
                volume_ratio = calculate_volume_ratio(core_box, shell_box)
                emulsion_boxes_list_copy.loc[i, 'volume_ratio'] = volume_ratio
    # 将参数中依然为0的值赋为空值
    emulsion_boxes_list_copy['diameter'] = emulsion_boxes_list_copy['diameter'].apply(lambda x: None if x == 0 else x)
    emulsion_boxes_list_copy['core_num'] = emulsion_boxes_list_copy['core_num'].apply(lambda x: None if x == 0 else x)
    emulsion_boxes_list_copy['concentricity'] = emulsion_boxes_list_copy['concentricity'].apply(
        lambda x: None if x == 0 else x)
    emulsion_boxes_list_copy['core_shell_ratio'] = emulsion_boxes_list_copy['core_shell_ratio'].apply(
        lambda x: None if x == 0 else x)
    emulsion_boxes_list_copy['volume_ratio'] = emulsion_boxes_list_copy['volume_ratio'].apply(
        lambda x: None if x == 0 else x)
    out_results = pd.DataFrame(columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                        'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    # 统计 diameter信息 ，首先是单乳液
    diameter_single = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 2]['diameter']
    diameter_single = pd.DataFrame(diameter_single, columns=['single diameter'])
    diameter_double = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 1]['diameter']
    diameter_double = pd.DataFrame(diameter_double, columns=['diameter'])
    # 查看数量
    if len(diameter_single) > 0:
        single_diameter = data_describe(diameter_single, 'single diameter')
        out_results = pd.concat([out_results, single_diameter], axis=0)
    if len(diameter_double) > 0:
        # 内液滴数量
        core_num = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 1]['core_num']
        core_num_describe = pd.DataFrame(core_num, columns=['core_num'])
        core_num_describe = data_describe(core_num_describe, 'core_num')
        # 壳数量
        double_shell_diameter = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 1]['diameter']
        double_shell_diameter = pd.DataFrame(double_shell_diameter, columns=['diameter'])
        double_shell_diameter = double_shell_diameter.rename(columns={'diameter': 'shell_diameter'})
        double_shell_diameter = data_describe(double_shell_diameter, 'shell_diameter')
        # 核数量
        data_core_diameter = pd.DataFrame(core_diameter, columns=['core_diameter'])
        double_core_diameter = data_describe(data_core_diameter, 'core_diameter')
        out_results = pd.concat([out_results, core_num_describe, double_shell_diameter, double_core_diameter], axis=0)
        # 同心度
        concentricity = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 1]['concentricity']
        concentricity = pd.DataFrame(concentricity, columns=['concentricity'])
        concentricity = concentricity.dropna()
        if len(concentricity) > 0:
            concentricity = data_describe(concentricity, 'concentricity')
            out_results = pd.concat([out_results, concentricity], axis=0)
        # 核壳比
        core_shell_ratio = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 1]['core_shell_ratio']
        core_shell_ratio = pd.DataFrame(core_shell_ratio, columns=['core_shell_ratio'])
        core_shell_ratio = core_shell_ratio.dropna()
        if len(core_shell_ratio) > 0:
            core_shell_ratio = data_describe(core_shell_ratio, 'core_shell_ratio')
            out_results = pd.concat([out_results, core_shell_ratio], axis=0)
        # 体积比
        volume_ratio = emulsion_boxes_list_copy[emulsion_boxes_list_copy['category_id'] == 1]['volume_ratio']
        volume_ratio = pd.DataFrame(volume_ratio, columns=['volume_ratio'])
        volume_ratio = volume_ratio.dropna()
        if len(volume_ratio) > 0:
            volume_ratio = data_describe(volume_ratio, 'volume_ratio')
            out_results = pd.concat([out_results, volume_ratio], axis=0)

    # ========液滴数量统计========

    # 查看每个emulsion_id 第一次出现时的ratio,如果ratio等于1 删除对应emulsion_id的所有行
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='ratio')

    emulsion_id_unique = emulsion_boxes_list['emulsion_id'].unique()
    for i in emulsion_id_unique:
        ratio = emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] == i]['ratio'].values[0]
        if ratio == 1:
            emulsion_boxes_list = emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] != i]
    # 查看每个emulsion_id 第一次出现时的ratio,如果ratio小于1，如果最后一次出现时的ratio小于1，且不是最后一帧，则让最后一帧的ratio等于1
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='ratio')
    emulsion_id_unique = emulsion_boxes_list['emulsion_id'].unique()
    for i in emulsion_id_unique:
        # 如果只有1帧，去掉
        if len(emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] == i]) < 2:
            emulsion_boxes_list = emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] != i]
            continue
        ratio_first = emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] == i]['ratio'].values[0]
        ratio_last = emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] == i]['ratio'].values[-1]

        if ratio_first < 1 and 0 < ratio_last < 1:
            emulsion_boxes_list.loc[emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] == i].index[-1], 'ratio'] = 1

    # 按emuision_id从小到大排序
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='emulsion_id')
    # 按image_id从小到大排序
    emulsion_boxes_list = emulsion_boxes_list.sort_values(by='image_id')
    # 重置索引
    emulsion_boxes_list = emulsion_boxes_list.reset_index(drop=True)
    emulsion_boxes_list['cell_type'] = 0
    emulsion_id_unique = emulsion_boxes_list['emulsion_id'].unique()
    # 将boxs列长度为8的行类别改为2， 其余为1
    for i in emulsion_id_unique:
        emulsion_info_now = emulsion_boxes_list[emulsion_boxes_list['emulsion_id'] == i]
        emulsion_info_now = emulsion_info_now.reset_index(drop=True)
        emulsion_info_now_length_list = []
        for j in range(len(emulsion_info_now)):
            emulsion_info_now_length = len(emulsion_info_now.loc[j, 'boxs'])
            emulsion_info_now_length_list.append(emulsion_info_now_length)
        emulsion_info_now_first = emulsion_info_now_length_list[0]
        emulsion_info_now_second = emulsion_info_now_length_list[1]
        if emulsion_info_now_first == 8 or emulsion_info_now_second == 8:
            emulsion_boxes_list.loc[emulsion_boxes_list['emulsion_id'] == i, 'cell_type'] = 2 # 2代表双液滴
        else:
            emulsion_boxes_list.loc[emulsion_boxes_list['emulsion_id'] == i, 'cell_type'] = 1 # 1代表其他
    # 计算每张图片已产生的液滴数量
    single_sum = []
    double_sum = []
    double_single_sum = []
    single_id = dict()
    double_id = dict()
    double_single_id = dict()
    image_id_unique = pkl_result['image_id'].unique()
    for i in range(len(image_id_unique)):
        emulsion_info_now = emulsion_boxes_list[emulsion_boxes_list['image_id'] == i]
        emulsion_info_now = emulsion_info_now.reset_index(drop=True)
        if emulsion_info_now.empty:
            single_sum.append(0)
            double_sum.append(0)
            double_single_sum.append(0)
            continue
        # 重置索引
        for j in range(len(emulsion_info_now)):
            # 查看当前液滴的编号
            emulsion_id = emulsion_info_now.loc[j, 'emulsion_id']
            emulsion_category_id = emulsion_info_now.loc[j, 'category_id']
            enulsion_cell_type = emulsion_info_now.loc[j, 'cell_type']
            emulsion_ratio = emulsion_info_now.loc[j, 'ratio']
            # 如果是单液滴
            if emulsion_category_id == 2:
                # 检查编号是否在single_id的字典中
                if emulsion_id in single_id:
                    # 如果在，则更新对应的值
                    single_id[emulsion_id] = emulsion_ratio
                else:
                    # 如果不在，说明是新的液滴，需要添加
                    single_id[emulsion_id] = emulsion_ratio
            # 如果是双液滴
            if emulsion_category_id == 1:
                # 检查编号是否在double_id的字典中
                if emulsion_id in double_id:
                    # 如果在，则更新对应的值
                    double_id[emulsion_id] = emulsion_ratio
                else:
                    # 如果不在，说明是新的液滴，需要添加
                    double_id[emulsion_id] = emulsion_ratio
            # 如果cell_type是2
            if enulsion_cell_type == 2:
                # 检查编号是否在double_single_id的字典中
                if emulsion_id in double_single_id:
                    # 如果在，则更新对应的值
                    double_single_id[emulsion_id] = emulsion_ratio
                else:
                    # 如果不在，说明是新的液滴，需要添加
                    double_single_id[emulsion_id] = emulsion_ratio
        single_sum.append(sum(single_id.values()))
        double_sum.append(sum(double_id.values()))
        double_single_sum.append(sum(double_single_id.values()))
    # 将single_sum和double_sum添加到emulsion_boxes_list中,并且要对应到image_id
    emulsion_boxes_list['single_sum'] = emulsion_boxes_list['image_id'].map(dict(zip(image_id_unique, single_sum)))
    emulsion_boxes_list['double_sum'] = emulsion_boxes_list['image_id'].map(dict(zip(image_id_unique, double_sum)))
    emulsion_boxes_list['double_single_sum'] = emulsion_boxes_list['image_id'].map(
        dict(zip(image_id_unique, double_single_sum)))
    # 提取image_id 和 single_sum 还有double_sum还有double_single_sum
    emulsion_img_single_sum_double = emulsion_boxes_list[['image_id', 'single_sum', 'double_sum', 'double_single_sum']]
    # 对于imageid重复的，只保留第一个
    emulsion_img_single_sum_double = emulsion_img_single_sum_double.drop_duplicates(subset='image_id', keep='first')
    # 查看第0帧的是否存在，如果没有，添加
    if 0 not in emulsion_img_single_sum_double['image_id'].values:
        emulsion_img_single_sum_double = pd.concat(
            [pd.DataFrame({'image_id': [0], 'single_sum': [0], 'double_sum': [0], 'double_single_sum': [0]}),
             emulsion_img_single_sum_double])
    # 查看image_id是否是由小到大 从1递增，如果中间有缺失，说明有漏帧，补齐
    image_id_unique = emulsion_img_single_sum_double['image_id'].unique()
    for i in range(1, len(image_id_unique)):
        if image_id_unique[i] - image_id_unique[i - 1] != 1:
            # 查看差几帧，进行插值补齐
            num = image_id_unique[i] - image_id_unique[i - 1]
            for j in range(1, num):
                emulsion_img_single_sum_double = pd.concat(
                    [emulsion_img_single_sum_double, pd.DataFrame({'image_id': [image_id_unique[i - 1] + j],
                                                                   'single_sum': [''], 'double_sum': ['']
                        , 'double_single_sum': ['']})])
        # 按image_id从小到大排序
    emulsion_img_single_sum_double = emulsion_img_single_sum_double.sort_values(by='image_id')
    emulsion_img_single_sum_double = emulsion_img_single_sum_double.reset_index(drop=True)
    # emulsion_img_single_sum_double用插值填补空值
    emulsion_img_single_sum_double['single_sum'] = emulsion_img_single_sum_double['single_sum'].replace('', np.nan)
    emulsion_img_single_sum_double['double_sum'] = emulsion_img_single_sum_double['double_sum'].replace('', np.nan)
    emulsion_img_single_sum_double['double_single_sum'] = emulsion_img_single_sum_double['double_single_sum'].replace(
        '', np.nan)
    emulsion_img_single_sum_double['single_sum'] = emulsion_img_single_sum_double['single_sum'].interpolate()
    emulsion_img_single_sum_double['double_sum'] = emulsion_img_single_sum_double['double_sum'].interpolate()
    emulsion_img_single_sum_double['double_single_sum'] = emulsion_img_single_sum_double['double_single_sum'].interpolate()
    emulsion_img_single_sum_double['single_sum'] = emulsion_img_single_sum_double['single_sum'].astype(float)
    emulsion_img_single_sum_double['double_sum'] = emulsion_img_single_sum_double['double_sum'].astype(float)
    emulsion_img_single_sum_double['double_single_sum'] = emulsion_img_single_sum_double['double_single_sum'].astype(float)
    # 重置索引
    emulsion_img_single_sum_double = emulsion_img_single_sum_double.reset_index(drop=True)
    single_one = emulsion_img_single_sum_double['single_sum'][0]
    double_one = emulsion_img_single_sum_double['double_sum'][0]
    double_single_one = emulsion_img_single_sum_double['double_single_sum'][0]
    emulsion_img_single_sum_double['single_sum'] = emulsion_img_single_sum_double['single_sum'].apply(
        lambda x: (float(x) - float(single_one)))
    emulsion_img_single_sum_double['double_sum'] = emulsion_img_single_sum_double['double_sum'].apply(
        lambda x: (float(x) - float(double_one)))
    emulsion_img_single_sum_double['double_single_sum'] = emulsion_img_single_sum_double['double_single_sum'].apply(
        lambda x: (float(x) - float(double_single_one)))
    # 确保数值是逐渐增加的
    for i in range(1, len(emulsion_img_single_sum_double)):
        if float(emulsion_img_single_sum_double.loc[i, 'single_sum']) - float(
                emulsion_img_single_sum_double.loc[i - 1, 'single_sum']) < 1e-6:
            emulsion_img_single_sum_double.loc[i, 'single_sum'] = emulsion_img_single_sum_double.loc[
                i - 1, 'single_sum']
        if float(emulsion_img_single_sum_double.loc[i, 'double_sum']) - float(
                emulsion_img_single_sum_double.loc[i - 1, 'double_sum']) < 1e-6:
            emulsion_img_single_sum_double.loc[i, 'double_sum'] = emulsion_img_single_sum_double.loc[
                i - 1, 'double_sum']
        if float(emulsion_img_single_sum_double.loc[i, 'double_single_sum']) - float(
                emulsion_img_single_sum_double.loc[i - 1, 'double_single_sum']) < 1e-6:
            emulsion_img_single_sum_double.loc[i, 'double_single_sum'] = emulsion_img_single_sum_double.loc[
                i - 1, 'double_single_sum']
    emulsion_img_single_sum_double['single_sum'] = emulsion_img_single_sum_double['single_sum'].astype(float)
    emulsion_img_single_sum_double['double_sum'] = emulsion_img_single_sum_double['double_sum'].astype(float)
    emulsion_img_single_sum_double['double_single_sum'] = emulsion_img_single_sum_double['double_single_sum'].astype(
        float)
    emulsion_img_single_sum_double['single_sum'] = emulsion_img_single_sum_double['single_sum'].apply(
        lambda x: '%.2f' % x)
    emulsion_img_single_sum_double['double_sum'] = emulsion_img_single_sum_double['double_sum'].apply(
        lambda x: '%.2f' % x)
    emulsion_img_single_sum_double['double_single_sum'] = emulsion_img_single_sum_double['double_single_sum'].apply(
        lambda x: '%.2f' % x)
    # 按image_id从小到大排序
    emulsion_img_single_sum_double = emulsion_img_single_sum_double.sort_values(by='image_id')
    emulsion_img_single_sum_double = emulsion_img_single_sum_double.reset_index(drop=True)
    # 查看单液滴数量
    print('single_sum:', emulsion_img_single_sum_double['single_sum'].values[-1])
    # 查看双液滴数量
    print('double_sum:', emulsion_img_single_sum_double['double_sum'].values[-1])
    # 查看双液滴1内液滴数量
    print('double_single_sum:', emulsion_img_single_sum_double['double_single_sum'].values[-1])
    # out_results添加新的一行
    new_data = pd.DataFrame(columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                     'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    # Variable name 赋值为core_num_single
    new_data.loc[0, 'Variable name'] = 'Double_single_count'
    # Count 赋值为double_single_sum
    new_data.loc[0, 'Count'] = float(emulsion_img_single_sum_double['double_single_sum'].values[-1])
    # 其他列的值为-1
    for col in ['Average value', 'Standard Deviation', 'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier']:
        new_data.loc[0, col] = -1
    out_results = pd.concat([out_results, new_data], axis=0)
    # out_results添加新的一行
    new_data = pd.DataFrame(columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                     'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    new_data.loc[0, 'Variable name'] = 'Single_count'
    new_data.loc[0, 'Count'] = float(emulsion_img_single_sum_double['single_sum'].values[-1])
    # 其他列的值为-1
    for col in ['Average value', 'Standard Deviation', 'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier']:
        new_data.loc[0, col] = -1
    out_results = pd.concat([out_results, new_data], axis=0)
    # out_results添加新的一行
    new_data = pd.DataFrame(columns=['Variable name', 'Count', 'Average value', 'Standard Deviation',
                                        'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier'])
    new_data.loc[0, 'Variable name'] = 'Double_count'
    new_data.loc[0, 'Count'] = float(emulsion_img_single_sum_double['double_sum'].values[-1])
    # 其他列的值为-1
    for col in ['Average value', 'Standard Deviation', 'Variance', 'Min', '25%', '50%', '75%', 'Max', 'outlier']:
        new_data.loc[0, col] = -1
    out_results = pd.concat([out_results, new_data], axis=0)
    # 保存视频
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(video_out_path, fourcc, fps, (video_width, video_height))
    # 添加标记
    from PIL import Image, ImageDraw, ImageFont
    font_path = "arial.ttf"
    image_id_unique = emulsion_img_single_sum_double['image_id'].unique()
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
    image_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取视频的高度
    font_size = int(image_height * (5 / 100))  # X为你希望字体占图片高度的百分比
    font = ImageFont.truetype(font_path, font_size)
    text_color = (255, 255, 255)
    frame_index = 0  # 初始化当前处理的帧编号
    pkl_result2 = read_pkl(pkl_path, nms_flag=True, short_long_ratio_flag=True, short_long_ratio_threshold=0.6)
    pkl_result2 = pd.DataFrame(pkl_result, columns=['image_id', 'category_id', 'bounding_box'])
    image_id_unique2 = pkl_result2['image_id'].unique()

    if statistical_mode == 'Cell encapsulation':
        text1 = 'Droplet with cell:'
        text2 = 'Droplet without cell:'
    elif statistical_mode == 'Single-cell encapsulated':
        text1 = 'Single-cell encapsulated:'
        text2 = 'Others:'
    elif statistical_mode == 'Double emulsion':
        text1 = 'Double emulsion droplets:'
        text2 = 'Single droplets:'
    elif statistical_mode == 'Single droplet':
        text1 = 'Double emulsion droplets:'
        text2 = 'Single droplets:'
    else:
        text1 = 'test_mode'
        text2 = 'test_mode'
    # 遍历每一帧
    while True:
        single_sum_font = str(0)
        double_sum_font = str(0)
        ret, frame = cap.read()
        if not ret:
            break  # 没有更多帧可读取时，结束循环
        # 如果当前帧号在 image_id_unique 中，进行处理
        if frame_index in image_id_unique:
            i = frame_index  # 当前帧号
            boxes_list_info = emulsion_img_single_sum_double[emulsion_img_single_sum_double['image_id'] == i]
            boxes_list_info = boxes_list_info.reset_index(drop=True)
            boxes_list = pkl_result[pkl_result['image_id'] == i]
            boxes_list = boxes_list.reset_index(drop=True)

            # 添加矩形框
            for j in range(len(boxes_list)):
                boxes = boxes_list.loc[j, 'bounding_box']
                category_id = boxes_list.loc[j, 'category_id']
                if category_id == 1:
                    color = (200, 20, 60)
                elif category_id == 2:
                    color = (0, 0, 235)
                else:
                    color = (0, 0, 0)
                x1, y1, x2, y2 = boxes[0:4]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 绘制判断线和其他处理...
            if Directional_significance == 0 or Directional_significance == 1:
                cv2.line(frame, (0, int(Judgment_line)), (video_width, int(Judgment_line)), (0, 255, 0), 2)
            elif Directional_significance == 2 or Directional_significance == 3:
                cv2.line(frame, (int(Judgment_line), 0), (int(Judgment_line), video_height), (0, 255, 0), 2)

            # 添加文字
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), "Example text", font=font)
            # 计算文字的宽度和高度
            text_width = bbox[2] - bbox[0]  # right - left
            text_height = bbox[3] - bbox[1]  # bottom - top
            # 定义行间距百分比，例如行间距为文字高度的20%
            line_spacing = int(text_height * 1.2)  # 增加20%的行间距
            if statistical_mode == 'Cell encapsulation':
                double_sum_font = str(boxes_list_info['double_sum'].values[0])
                single_sum_font = str(boxes_list_info['single_sum'].values[0])
                draw.text((10, 10), text1 + double_sum_font, text_color, font=font)
                draw.text((10, 10+line_spacing), text2 + single_sum_font, text_color, font=font)
            elif statistical_mode == 'Double emulsion':
                double_sum_font = str(boxes_list_info['double_sum'].values[0])
                single_sum_font = str(boxes_list_info['single_sum'].values[0])
                draw.text((10, 10), text1 + double_sum_font, text_color, font=font)
                draw.text((10, 10+line_spacing), text2 + single_sum_font, text_color, font=font)
            elif statistical_mode == 'Single-cell encapsulated':
                double_sum_font = str(boxes_list_info['double_single_sum'].values[0])
                single_sum_font = (float(boxes_list_info['single_sum'].values[0]) + float(boxes_list_info['double_sum'].values[0])
                                   - float(boxes_list_info['double_single_sum'].values[0]))
                # single_sum_font保留两位小数
                single_sum_font = '%.2f' % single_sum_font
                single_sum_font = str(single_sum_font)
                draw.text((10, 10), text1 + double_sum_font, text_color, font=font)
                draw.text((10, 10+line_spacing), text2 + single_sum_font, text_color, font=font)
            elif statistical_mode == 'Single droplet':
                double_sum_font = str(boxes_list_info['double_sum'].values[0])
                single_sum_font = str(boxes_list_info['single_sum'].values[0])
                draw.text((10, 10), text2 + double_sum_font, text_color, font=font)
                draw.text((10, 10+line_spacing), text2 + single_sum_font, text_color, font=font)

            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        elif frame_index in image_id_unique2:
            i = frame_index
            boxes_list = pkl_result2[pkl_result2['image_id'] == i]
            boxes_list = boxes_list.reset_index(drop=True)
            # 添加矩形框
            for j in range(len(boxes_list)):
                boxes = boxes_list.loc[j, 'bounding_box']
                category_id = boxes_list.loc[j, 'category_id']
                if category_id == 1:
                    color = (200, 20, 60)
                elif category_id == 2:
                    color = (0, 0, 235)
                else:
                    color = (0, 0, 0)
                x1, y1, x2, y2 = boxes[0:4]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                color, 2)
            # 添加判断线
            if Directional_significance == 0 or Directional_significance == 1:
                cv2.line(frame, (0, int(Judgment_line)), (video_width, int(Judgment_line)), (0, 255, 0), 2)
            elif Directional_significance == 2 or Directional_significance == 3:
                cv2.line(frame, (int(Judgment_line), 0), (int(Judgment_line), video_height), (0, 255, 0), 2)
            # 添加文字
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            bbox = draw.textbbox((0, 0), "Example text", font=font)
            # 计算文字的宽度和高度
            text_width = bbox[2] - bbox[0]  # right - left
            text_height = bbox[3] - bbox[1]  # bottom - top
            # 定义行间距百分比，例如行间距为文字高度的20%
            line_spacing = int(text_height * 1.2)  # 增加20%的行间距
            if statistical_mode == 'Cell encapsulation':
                draw.text((10, 10 ), text1 + double_sum_font, text_color, font=font)
                draw.text((10, 10+ line_spacing), text2 + single_sum_font, text_color, font=font)
            elif statistical_mode == 'Double emulsion':
                draw.text((10, 10), text1 + double_sum_font, text_color, font=font)
                draw.text((10, 10 + line_spacing), text2 + single_sum_font, text_color, font=font)
            elif statistical_mode == 'Single-cell encapsulated':
                draw.text((10, 10 ), text1 + double_sum_font, text_color, font=font)
                draw.text((10, 10+ line_spacing), text2 + single_sum_font, text_color, font=font)
            elif statistical_mode == 'Single droplet':
                draw.text((10, 10 ), text2 + double_sum_font, text_color, font=font)
                draw.text((10, 10+ line_spacing), text2 + single_sum_font, text_color, font=font)
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        # 将帧写入输出视频，即使该帧未处理
        out.write(frame)
        # 递增帧编号
        frame_index += 1
    # 保留两位小数
    out_results = out_results.round(2)

    return out_results, video_out_path


if __name__ == '__main__':
    video_statistical(video_path=r'C:\ML\emulsion\backend\services\test_data\test_video.mp4',
                      pkl_path=r'C:\ML\emulsion\backend\services\test_data\test_video\out\faster_voc.pkl',
                      video_out_path=r'C:\ML\emulsion\backend\services\test_data\test_video_out.mp4',
                      statistical_mode='Single droplet', time_span=1, flow_direction='left-to-right', reference_line=80)
