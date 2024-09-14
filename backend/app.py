from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from werkzeug.utils import secure_filename
import os
from services.image_untils import image_recognition
from services.video_until import video_recognition
from flask_cors import CORS
import logging
from datetime import datetime
import json
from werkzeug.exceptions import RequestEntityTooLarge
from datetime import datetime
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi'}
    # os.path.splitext(filename) 返回一个元组，第一个元素是文件名，第二个是扩展名
    ext = os.path.splitext(filename)[1].lower()
    # 只处理文件名中包含扩展名的情况
    print(ext)
    return ext != '' and ext[1:] in ALLOWED_EXTENSIONS

# 设置错误处理器

def setup_routes(app):
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_size_exceed(error):
        return jsonify({'error': 'File too large. Maximum allowed size is 10MB.'}), 413

    @app.errorhandler(400)
    def handle_bad_request(error):
        return jsonify({'error': 'Bad request. Please check your parameters.'}), 400

    @app.errorhandler(500)
    def handle_internal_error(error):
        logging.error(f"Internal server error: {str(error)}", exc_info=True)
        return jsonify({'error': 'An internal error occurred. Please try again later.'}), 500
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                raise ValueError('No file part')
            file = request.files['file']
            if file.filename == '':
                raise ValueError('No selected file')
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                # 获取当前时间，格式化为字符串
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                # 为文件名添加时间戳，格式为: 原文件名_时间戳.扩展名
                name, ext = os.path.splitext(filename)
                filename = f"{name}_{timestamp}{ext}"

                # 保存文件到指定文件夹
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                return jsonify({'message': 'File successfully uploaded', 'file_path': file_path}), 200
            else:
                raise ValueError('File type not allowed')
        except ValueError as e:
            logging.error(f"Value error during file upload: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logging.error(f"Unexpected error during file upload: {str(e)}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred. Please try again later.'}), 500

    @app.route('/ImageRecognize/<filename>', methods=['GET'])
    def get_result_file(filename):
        try:
            return send_from_directory(app.config['RESULT_FOLDER'], filename)
        except Exception as e:
            logging.error(f"Error sending file: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 400
    @app.route('/ImageRecognize/<path:filename>', methods=['GET'])
    def get_image(filename):
        try:
            return send_from_directory(app.config['RESULT_FOLDER'], filename)
        except Exception as e:
            logging.error(f"Error sending image: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 400

    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(app.static_folder + '/' + path):
            return send_from_directory(app.static_folder, path)
        else:
            return render_template('index.html')

    @app.route('/api/video_visualize', methods=['POST'])
    def video_visualize():
        try:
            data = request.get_json()
            actual_size = data.get('actualSize')
            unit = data.get('unit')
            timeSpan = data.get('timeSpan')
            flowDirection = data.get('flowDirection')
            referenceLine = data.get('referenceLine')
            selectedmode = data.get('selectedMode')
            selectedAlgorithm = data.get('selectedAlgorithm')
            params = {
                'actual_size': actual_size,
                'unit': unit,
                'timeSpan': timeSpan,
                'flowDirection': flowDirection,
                'referenceLine': referenceLine,
                'selectedmode': selectedmode,
                'selectedAlgorithm': selectedAlgorithm
            }
            logging.info('User input parameters: %s', json.dumps(params))
            # 处理图像并获取结果路径和表格数据
            filename = data.get('filename')
            # 获取没有后缀的文件名
            results, video_name, image_names = video_recognition(filename, app.config['RESULT_FOLDER'] + '/',
                                                                 actual_size=actual_size,
                                                                 actual_unit=unit
                                                                 , time_span=timeSpan, flow_direction=flowDirection,
                                                                 reference_line=referenceLine, statistical_mode=selectedmode
                                                                 , algorithm=selectedAlgorithm)
            # image_names 去除最顶级目录
            image_names = [name.split('/')[-3] + '/' + name.split('/')[-2] + '/' + name.split('/')[-1] for name in
                           image_names]
            # 构建图片的访问URL
            image_names = [url_for('get_result_file', filename=name, _external=True) for name in image_names]
            results = results.fillna('null')
            withcount, withoutcount = getwithcount_video(results, unit, selectedmode)
            # 保留两位小数
            withcount = round(withcount, 2)
            withoutcount = round(withoutcount, 2)
            # 去除Double_single_count、Double_count、Single_count行
            results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
            # 根据模式保留对应的行
            mode_double = ['core_num', 'shell_diameter_' + unit, 'core_diameter_' + unit, 'single_diameter_' + unit,
                           'concentricity', 'core_shell_ratio', 'volume_ratio']
            model_single = ['single_diameter_' + unit]
            model_cell = ['shell_diameter_' + unit]
            if selectedmode == 'Double emulsion':
                # 保留双液滴模式的行，如果有
                results = results[results['Variable name'].isin(mode_double)]
            elif selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
                results = results[results['Variable name'].isin(model_cell)]
            elif selectedmode == 'Single droplet':
                # 保留单液滴模式的行，如果有
                results = results[results['Variable name'].isin(model_single)]
            data = results.values.tolist()
            video_name = video_name.split('/')[-2] + '/' + video_name.split('/')[-1]
            # 构建视频的访问URL
            video_path = url_for('get_result_file', filename=video_name, _external=True)
            logging.info('video_path:', video_path)
            # 返回两个处理后的图片路径和表格数据
            return jsonify({'video_path': video_path, 'withcount': withcount,
                            'withoutcount': withoutcount, 'image_names': image_names}), 200
        except ValueError as e:
            logging.error(f"Value error: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred during video processing.'}), 500

    @app.route('/api/video_statistics', methods=['POST'])
    def video_statistics():
        try:
            # 从请求中获取前端传递的参数
            data = request.get_json()
            actual_size = data.get('actualSize')
            unit = data.get('unit')
            timeSpan = data.get('timeSpan')
            flowDirection = data.get('flowDirection')
            referenceLine = data.get('referenceLine')
            selectedmode = data.get('selectedMode')
            selectedalgorithm = data.get('selectedAlgorithm')
            params = {
                'actual_size': actual_size,
                'unit': unit,
                'timeSpan': timeSpan,
                'flowDirection': flowDirection,
                'referenceLine': referenceLine,
                'selectedmode': selectedmode,
                'selectedAlgorithm': selectedalgorithm
            }
            logging.info('User input parameters: %s', json.dumps(params))
            # 假设已经上传并保存了图像文件
            filename = data.get('filename')
            print(filename)
            # 调用图像处理函数，仅返回统计数据
            results, _, image_names = video_recognition(filename, app.config['RESULT_FOLDER'] + '/',
                                                        actual_size=actual_size,
                                                        actual_unit=unit
                                                        , time_span=timeSpan, flow_direction=flowDirection,
                                                        reference_line=referenceLine,
                                                        statistical_mode=selectedmode,
                                                        algorithm=selectedalgorithm)
            results = results.fillna('null')
            withcount, withoutcount = getwithcount_video(results, unit, selectedmode)
            withcount = round(withcount, 2)
            withoutcount = round(withoutcount, 2)
            # 去除Double_single_count、Double_count、Single_count行
            results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
            # 根据模式保留对应的行
            mode_double = ['core_num', 'shell_diameter_' + unit, 'core_diameter_' + unit, 'single_diameter_' + unit,
                           'concentricity', 'core_shell_ratio', 'volume_ratio']
            model_single = ['single_diameter_' + unit]
            model_cell = ['shell_diameter_' + unit]
            if selectedmode == 'Double emulsion' :
                # 保留双液滴模式的行，如果有
                results = results[results['Variable name'].isin(mode_double)]
            elif selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
                # 保留单液滴模式的行，如果有
                results = results[results['Variable name'].isin(model_cell)]
            elif selectedmode == 'Single droplet':
                # 保留单液滴模式的行，如果有
                results = results[results['Variable name'].isin(model_single)]
            # 确保四舍五入操作仅对数值类型的列进行
            numeric_columns = results.select_dtypes(include=['float64', 'int64']).columns
            results[numeric_columns] = results[numeric_columns].round(2)
            # 如果Variable name中有后三位是_ + actual_unit的行,替换为 (actual_unit)
            for i in range(len(results['Variable name'].values)):
                if results['Variable name'].values[i][-3:] == '_' + unit:
                    results.iloc[i, 0] = results.iloc[i, 0][:-3] + ' (' + unit + ')'
            data = results.values.tolist()
            columns = results.columns.tolist()
            # 合并列名和数据
            stata_data = [columns] + data
            # 返回统计数据
            return jsonify({'statistics': stata_data, 'withcount': withcount, 'withoutcount': withoutcount}), 200
            # 构建返回结果...
        except ValueError as e:
            logging.error(f"Value error: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({'error': 'An unexpected error occurred during video processing.'}), 500

    @app.route('/api/img_visualize', methods=['POST'])
    def img_visualize():
        data = request.get_json()
        actual_size = data.get('actualSize')
        unit = data.get('unit')
        selectedmode = data.get('selectedMode')
        selectedalogrithm = data.get('selectedAlgorithm')
        params = {
            'actual_size': actual_size,
            'unit': unit,
            'selectedmode': selectedmode,
            'selectedAlgorithm': selectedalogrithm
        }
        logging.info('User input parameters: %s', json.dumps(params))
        # 处理图像并获取结果路径和表格数据
        filename = data.get('filename')
        # 获取没有后缀的文件名
        results, image_names = image_recognition(filename, app.config['RESULT_FOLDER'] + '/',
                                                 actual_size=actual_size, actual_unit=unit,
                                                 statistical_mode=selectedmode,
                                                 algorithm=selectedalogrithm)
        # 将results中的Nan替换为null
        results = results.fillna('null')
        withcount, withoutcount = getwithcount(results, unit, selectedmode)
        # 去除Double_single_count、Double_count、Single_count行
        results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
        # 根据模式保留对应的行
        mode_double = ['core_num', 'shell_diameter_' + unit, 'core_diameter_' + unit, 'single_diameter_' + unit,
                       'concentricity', 'core_shell_ratio', 'volume_ratio']
        model_single = ['single_diameter_' + unit]
        if selectedmode == 'Double emulsion' or selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
            # 保留双液滴模式的行，如果有
            results = results[results['Variable name'].isin(mode_double)]
        elif selectedmode == 'Single droplet':
            # 保留单液滴模式的行，如果有
            results = results[results['Variable name'].isin(model_single)]
        data = results.values.tolist()
        # image_names 去除最顶级目录
        image_names = [name.split('/')[-2] + '/' + name.split('/')[-1] for name in image_names]
        # 构建图片的访问URL
        image_names = [url_for('get_result_file', filename=name, _external=True) for name in image_names]
        # 将results转换为字典列表格式
        data = results.values.tolist()
        columns = results.columns.tolist()
        # 合并列名和数据
        stata_data = [columns] + data
        # 返回两个处理后的图片路径和表格数据
        return jsonify({'image_names': image_names, 'statistics': stata_data, 'withcount': withcount,
                        'withoutcount': withoutcount}), 200

    @app.route('/api/img_statistics', methods=['POST'])
    def img_statistics():
        # 从请求中获取前端传递的参数
        data = request.get_json()
        actual_size = data.get('actualSize')
        unit = data.get('unit')
        selectedmode = data.get('selectedMode')
        selectedAlgorithm = data.get('selectedAlgorithm')
        params = {
            'actual_size': actual_size,
            'unit': unit,
            'selectedmode': selectedmode,
            'selectedAlgorithm': selectedAlgorithm
        }
        logging.info('User input parameters: %s', json.dumps(params))
        # 假设已经上传并保存了图像文件
        filename = data.get('filename')
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # 调用图像处理函数，仅返回统计数据
        results, _ = image_recognition(filename, app.config['RESULT_FOLDER'] + '/', actual_size=actual_size,
                                       actual_unit=unit,
                                       statistical_mode=selectedmode, algorithm=selectedAlgorithm)
        # 将results中的Nan替换为null
        results = results.fillna('null')
        withcount, withoutcount = getwithcount(results, unit, selectedmode)
        # 去除Double_single_count、Double_count、Single_count行
        results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
        # 根据模式保留对应的行
        mode_double = ['core_num', 'shell_diameter_' + unit, 'core_diameter_' + unit, 'single_diameter_' + unit,
                       'concentricity', 'core_to_shell_ratio', 'volume_ratio']
        model_single = ['single_diameter_' + unit]
        if selectedmode == 'Double emulsion' or selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
            # 保留双液滴模式的行，如果有
            results = results[results['Variable name'].isin(mode_double)]
        elif selectedmode == 'Single droplet':
            # 保留单液滴模式的行，如果有
            results = results[results['Variable name'].isin(model_single)]
       # 如果Variable name中有后三位是_ + actual_unit的行,替换为 (actual_unit)
        for i in range(len(results['Variable name'].values)):
            if results['Variable name'].values[i][-3:] == '_' + unit:
                results.iloc[i, 0] = results.iloc[i, 0][:-3] + ' (' + unit + ')'
        data = results.values.tolist()
        columns = results.columns.tolist()
        # 合并列名和数据
        stata_data = [columns] + data
        print(stata_data)
        # 返回统计数据
        return jsonify({'statistics': stata_data, 'withcount': withcount, 'withoutcount': withoutcount}), 200
def getwithcount(results, actual_unit, selectedmode):
    # 单液滴数量为relusts中single_diameter + ‘_’ + actual_unit行中的Count列的值
    if 'single_diameter_' + actual_unit in results['Variable name'].values:
        single_droplet_count = results[results['Variable name'] == 'single_diameter_' + actual_unit]['Count'].values[0]
    else:
        single_droplet_count = 0
    # 双液滴数量为relusts中shell_diameter + ‘_’ + actual_unit行中的Count列的值
    if 'shell_diameter_' + actual_unit in results['Variable name'].values:
        double_droplet_count = results[results['Variable name'] == 'shell_diameter_' + actual_unit]['Count'].values[0]
    else:
        double_droplet_count = 0
    # 只包含一个液滴的双液滴数量为relusts中core_num_single + ‘_’ + actual_unit行中的Count列的值
    if 'core_num_single' in results['Variable name'].values:
        double_single_droplet_count = results[results['Variable name'] == 'core_num_single']['Count'].values[0]
    else:
        double_single_droplet_count = 0

    if selectedmode == 'Double emulsion':
        withcount = double_droplet_count
        withoutcount = single_droplet_count
    elif selectedmode == 'Single droplet':
        withcount = single_droplet_count
        withoutcount = single_droplet_count
    elif selectedmode == 'Cell encapsulation':
        withcount = double_droplet_count
        withoutcount = single_droplet_count
    elif selectedmode == 'Single-cell encapsulated':
        withcount = double_single_droplet_count
        withoutcount = single_droplet_count + double_droplet_count - double_single_droplet_count
    else:
        withcount = 0
        withoutcount = 0
    return withcount, withoutcount


def getwithcount_video(results, actual_unit, selectedmode):
    # 单液滴数量为relusts中single_diameter + ‘_’ + actual_unit行中的Count列的值
    single_droplet_count = results[results['Variable name'] == 'Single_count']['Count'].values[0]
    # 双液滴数量为relusts中shell_diameter + ‘_’ + actual_unit行中的Count列的值
    double_droplet_count = results[results['Variable name'] == 'Double_count']['Count'].values[0]
    # 只包含一个液滴的双液滴数量为relusts中core_num_single + ‘_’ + actual_unit行中的Count列的值
    double_single_droplet_count = results[results['Variable name'] == 'Double_single_count']['Count'].values[0]

    if selectedmode == 'Double emulsion':
        withcount = double_droplet_count
        withoutcount = single_droplet_count
    elif selectedmode == 'Single droplet':
        withcount = single_droplet_count
        withoutcount = single_droplet_count
    elif selectedmode == 'Cell encapsulation':
        withcount = double_droplet_count
        withoutcount = single_droplet_count
    elif selectedmode == 'Single-cell encapsulated':
        withcount = double_single_droplet_count
        withoutcount = single_droplet_count + double_droplet_count - double_single_droplet_count
    else:
        withcount = 0
        withoutcount = 0
    return withcount, withoutcount

# 日志记录
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_file_path = os.path.join('logs', log_filename)
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]',
                    handlers=[logging.FileHandler(log_file_path, encoding='utf-8'), logging.StreamHandler()])

app = Flask(__name__, static_folder='dist', template_folder='dist')
setup_routes(app)
CORS(app)
# 设置上传文件夹和结果文件夹
UPLOAD_FOLDER = 'uploads'  # 上传文件夹
RESULT_FOLDER = 'ImageRecognize'  # 结果文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # 确保上传文件夹存在
os.makedirs(RESULT_FOLDER, exist_ok=True) # 确保结果文件夹存在

app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 设置上传文件夹
app.config['RESULT_FOLDER'] = RESULT_FOLDER  # 设置结果文件夹


if __name__ == '__main__':
    # app.run(debug=True)
    # app.run()
    app.run(host="0.0.0.0", port=8080)
