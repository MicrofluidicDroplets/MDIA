import mmcv
import pandas as pd
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import numpy as np
from tqdm import tqdm
import sys
import cv2
import os
from matplotlib import pyplot as plt
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from evaluation_object_metrics import video_statistical


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
    print('视频分解完成！')
    return fps, frame_width, frame_height


def video_recognition(video_path, out_dir,
                      actual_size=1, actual_unit='mm',algorithm=None, statistical_mode='double_emulsion',
                                time_span=1,flow_direction='left-to-right', reference_line=50):
    video_name = os.path.basename(video_path)
    pkl_name = video_name.split('.')[0] + '.pkl'
    out_dir = out_dir + video_name.split('.')[0] + '/'
    print(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_fps, frame_width, frame_height = video_breakdown(video_path, out_dir)
    if statistical_mode == 'Single droplet' and algorithm == 'Edge detection':
        # 读取图片
        imgs = []
        results = []  # 存储当前图像中所有矩形的坐标
        results_sum = []  # 最终结果的格式
        for img in os.listdir(out_dir):
            if img.endswith('.jpg') or img.endswith('.png'):
                imgs.append(os.path.join(out_dir, img))
        # imgs 按数字排序
        imgs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        results = []
        for img in tqdm(imgs, colour='green', desc='图片识别中'):
            img = cv2.imread(img,cv2.IMREAD_COLOR)
            edges = cv2.Canny(img, 5, 250) # 边缘检测 5是低阈值，250是高阈值
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
            results = []  # 存储当前图像中所有矩形的坐标
            results_sum = []  # 最终结果的格式
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                ((cx, cy), radius) = cv2.minEnclosingCircle(contour)
                if len(approx) >= 5 and cv2.contourArea(contour) > 200: # 获取外接矩形,approx是近似多边形的顶点,contourArea是计算轮廓面积
                    # 获取外接矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    x1, y1 = x, y  # 左上角坐标
                    x2, y2 = x + w, y + h  # 右下角坐标
                    # 保存坐标到列表 (x1, y1, x2, y2, 0)
                    results.append(np.array([x1, y1, x2, y2, 1]))
            results_sum.append([[], results, []])
        mmcv.dump(results_sum, out_dir + pkl_name)
    else:
        config_file = current_dir + r'\my_deformable_detr_r50_16x2_50e_coco.py'
        checkpoint_file =current_dir + r'\latest.pth'
        model = init_detector(config_file, checkpoint_file)
        imgs = []
        for img in os.listdir(out_dir):
            if img.endswith('.jpg') or img.endswith('.png'):
                imgs.append(os.path.join(out_dir, img))
        # imgs 按数字排序
        imgs.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
        results = []
        for img in tqdm(imgs, colour='green', desc='图片识别中'):
            result = inference_detector(model, img)
            results.append(result)
        mmcv.dump(results, out_dir + pkl_name)
    print(video_path, out_dir + pkl_name, out_dir + video_name, statistical_mode)
    results,_ = video_statistical(video_path, out_dir + pkl_name, out_dir + video_name, statistical_mode=statistical_mode,
                                time_span=time_span,flow_direction=flow_direction, reference_line=reference_line)
    # 单位转换，如果存在single_diameter，core_diameter，shell_diameter，则转换为实际尺寸
    actual_variable = ['single_diameter', 'core_diameter', 'shell_diameter']
    for i in range(len(results['Variable name'].values)):
        # 如果 Variable name列中包含actual_variable中的元素，则进行单位转换
        if results['Variable name'].values[i] in actual_variable:
            # 从第3列开始进行单位转换
            for j in range(2, len(results.columns)-1):
                results.iloc[i, j] = results.iloc[i, j] * float(actual_size)
            # 将名称改为带单位的名称
            results.iloc[i, 0] = str(results.iloc[i, 0]) + '_' + actual_unit
            ## 最后一列为列表，如果不为空，则进行单位转换
            if results.iloc[i, -1] != None and len(results.iloc[i, -1]) > 0:
                for k in range(len(results.iloc[i, -1])):
                    results.iloc[i, -1][k] = results.iloc[i, -1][k] * float(actual_size)
    # 定义颜色和格式
    boxprops = dict(color="blue", linewidth=2, facecolor='lightblue')  # 设置箱体的填充色
    flierprops = dict(marker='o', color='red', alpha=0.5)  # 异常值的样式
    medianprops = dict(color='green', linewidth=2)
    whiskerprops = dict(color='black', linewidth=1.5)
    capprops = dict(color='black', linewidth=1.5)
    image_names = []
    # 绘制箱线图
    for index, row in results.iterrows():
        mode_double = ['core_num', 'shell_diameter_' + actual_unit, 'core_diameter_' + actual_unit, 'single_diameter_' + actual_unit,
                       'concentricity', 'core_shell_ratio', 'volume_ratio']
        model_single = ['single_diameter_' + actual_unit]
        if statistical_mode == 'Cell encapsulation' and row['Variable name'] in mode_double:
                pass
        elif statistical_mode == 'Single-cell encapsulated'and row['Variable name'] in mode_double:
                pass
        elif statistical_mode == 'Double emulsion' and row['Variable name'] in mode_double:
                pass
        elif statistical_mode == 'Single droplet' and row['Variable name'] in model_single:
                pass
        else:
            continue
        # 如果count小于10，则不绘制箱线图
        if row['Count'] < 10:
            continue
        plt.figure(figsize=(10, 6))
        # 计算 IQR 和须线位置
        IQR = row['75%'] - row['25%']
        lower_whisker = max(row['Min'], row['25%'] - 1.5 * IQR)
        upper_whisker = min(row['Max'], row['75%'] + 1.5 * IQR)

        # 准备箱线图的数据
        box_data = [lower_whisker, row['25%'], row['50%'], row['75%'], upper_whisker]

        # 绘制箱线图
        plt.boxplot(
            [box_data],
            labels=[row['Variable name']],
            boxprops=boxprops, flierprops=flierprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops, patch_artist=True  # 启用填充颜色
        )

        # 绘制异常值
        if row['outlier'] is not None:
            plt.scatter([1] * len(row['outlier']), row['outlier'], color='red', alpha=0.6, marker='o')

        # 添加标题和标签
        plt.title(f'Boxplot for {row["Variable name"]}', fontsize=14, fontweight='bold')
        plt.xlabel('Variable', fontsize=12)
        plt.ylabel('Values', fontsize=12)
        if os.path.exists(out_dir + 'out/') == False:
            os.makedirs(out_dir + 'out/')
        plt.savefig(out_dir + 'out/' + row['Variable name'] + '.png')
        image_names.append(out_dir + 'out/' + row['Variable name'] + '.png')
    results = results.round(2)
    print(results)
    # 判断video_name后缀是否为.mp4 如果不是，则替换为.mp4
    if video_name.split('.')[-1] != 'mp4':
        video_name = video_name.split('.')[0] + '.mp4'
    return results,out_dir + video_name,image_names
if __name__ == '__main__':
    video_recognition('C:/ML/emulsion/backend/services/test_data/test_video2.mp4', 'C:/ML/emulsion/backend/services/')