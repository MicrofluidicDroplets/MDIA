import cv2
import os
import numpy as np
import pandas as pd

# Edge detection
# 单张输入图像路径
image_path = r'C:\ML\emulsion\backend\services\test_data\double.jpg'  # 修改为你要处理的图片路径
output_path = r'C:\ML\emulsion\backend\services\test_data\_output/'  # 输出路径
pkl_output_path = os.path.join(output_path, 'faster_voc_single.pkl')  # 保存坐标的pkl文件路径
csv_output_path = os.path.join(output_path, 'diameter_single.csv')
min_area = 10  # 最小面积阈值
max_area = 10000  # 最大面积阈值
min_radius = 1  # 最小半径阈值
max_radius = 100  # 最大半径阈值
min_aspect_ratio = 0.6  # 最小宽高比
max_aspect_ratio = 1.4  # 最大宽高比

# 用于保存坐标的字典
rect_coords_dict = {}
diameter_data = []
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 读取图像
filename = os.path.basename(image_path)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
edges = cv2.Canny(image, 5, 250)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = np.zeros_like(image)

results = []  # 存储当前图像中所有矩形的坐标
results_sum = []  # 最终结果的格式

for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    ((cx, cy), radius) = cv2.minEnclosingCircle(contour)

    if len(approx) >= 5 and cv2.contourArea(contour) > 200:
        # 获取外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1 = x, y  # 左上角坐标
        x2, y2 = x + w, y + h  # 右下角坐标

        # 绘制红色矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色矩形框

        # 保存坐标到列表 (x1, y1, x2, y2, 0)
        results.append([x1, y1, x2, y2, 0])

# 包装成期望的格式 ([], results, [])
results_sum.append(([], results, []))

print(results_sum)

# 保存处理后的图像
cv2.imwrite(os.path.join(output_path, filename), image)
