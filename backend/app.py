from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import os
import mmcv
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector,show_result_pyplot
import pandas as pd
app = Flask(__name__, static_folder='dist', template_folder='dist')

# 设置上传文件夹和允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'ImageRecognize'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 确保上传文件夹和结果文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 调用你的算法处理图片
        result_path = process_image(file_path, filename)

        return jsonify({'message': 'File successfully uploaded and processed', 'result_path': result_path}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/results/<filename>', methods=['GET'])
def get_result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return render_template('index.html')


def process_image(file_path, filename):
    config_file = 'my_deformable_detr_r50_16x2_50e_coco.py'
    checkpoint_file = 'latest.pth'
    model = init_detector(config_file, checkpoint_file)
    result = inference_detector(model, file_path)
    # 处理后的图片保存路径
    result_path = app.config['RESULT_FOLDER'] + '/' + filename
    show_result_pyplot(model, file_path, result, score_thr=0.5, out_file=result_path)
    # 返回处理后图片的相对路径
    return filename


if __name__ == '__main__':
    app.run(debug=True)
