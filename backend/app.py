from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from werkzeug.utils import secure_filename
import os
from services.image_untils import image_recognition
from services.video_until import video_recognition
from flask_cors import CORS

app = Flask(__name__, static_folder='dist', template_folder='dist')
CORS(app)
UPLOAD_FOLDER = 'uploads'  # 上传文件夹
RESULT_FOLDER = 'ImageRecognize'  # 结果文件夹
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi'}  # 允许上传的文件类型
app.config['MAX_CONTENT_LENGTH'] = 200 * 4096 * 4096  # 限制上传文件的大小
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 设置上传文件夹
app.config['RESULT_FOLDER'] = RESULT_FOLDER  # 设置结果文件夹

# 确保上传文件夹和结果文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)
def getwithcount(results, actual_unit,selectedmode):
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
def getwithcount_video(results, actual_unit,selectedmode):
    # 单液滴数量为relusts中single_diameter + ‘_’ + actual_unit行中的Count列的值
    single_droplet_count = results[results['Variable name'] == 'Single_count']['Count'].values[0]
    # 双液滴数量为relusts中shell_diameter + ‘_’ + actual_unit行中的Count列的值
    double_droplet_count = results[results['Variable name'] == 'Double_count' ]['Count'].values[0]
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
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/upload', methods=['POST'])
def upload_file():
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # 如果用户没有选择文件，浏览器也会发送一个空的文件名
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # 如果文件存在并且是允许的文件类型
    if file and allowed_file(file.filename):  # 判断文件类型是否允许上传
        filename = secure_filename(file.filename)  # 获取文件名
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # 保存文件
        file.save(file_path)  # 保存文件
        return jsonify({'message': 'File successfully uploaded', 'file_path': file_path}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/ImageRecognize/<filename>', methods=['GET'])
def get_result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


@app.route('/ImageRecognize/<path:filename>', methods=['GET'])
def get_image(filename):
    full_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    print(f"Serving file from: {full_path}")  # 打印实际的文件路径
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


# 为了支持前端路由，需要添加以下代码
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return render_template('index.html')


@app.route('/api/video_visualize', methods=['POST'])
def video_visualize():
    data = request.get_json()
    actual_size = data.get('actualSize')
    unit = data.get('unit')
    timeSpan = data.get('timeSpan')
    flowDirection = data.get('flowDirection')
    referenceLine = data.get('referenceLine')
    selectedmode = data.get('selectedMode')
    selectedAlgorithm = data.get('selectedAlgorithm')
    print('输入参数')
    print('actual_size:', actual_size)
    print('unit:', unit)
    print('timeSpan:', timeSpan)
    print('flowDirection:', flowDirection)
    print('referenceLine:', referenceLine)
    print('selectedmode:', selectedmode)
    print('selectedAlgorithm:', selectedAlgorithm)
    # 处理图像并获取结果路径和表格数据
    filename = data.get('filename')
    # 获取没有后缀的文件名
    results, video_name,image_names = video_recognition(filename, app.config['RESULT_FOLDER'] + '/', actual_size = actual_size,
                                                        actual_unit = unit
                                            , time_span=timeSpan, flow_direction=flowDirection,
                                            reference_line=referenceLine, statistical_mode=selectedmode
                                            , algorithm=selectedAlgorithm)
    # image_names 去除最顶级目录
    image_names = [name.split('/')[-3] + '/' + name.split('/')[-2] + '/' + name.split('/')[-1] for name in image_names]
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
    mode_double = ['core_num','shell_diameter_'+unit,'core_diameter_'+unit,'single_diameter_'+unit,
                   'concentricity','core_shell_ratio','volume_ratio']
    model_single = ['single_diameter_'+unit]
    if selectedmode == 'Double emulsion' or selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
        # 保留双液滴模式的行，如果有
        results = results[results['Variable name'].isin(mode_double)]
    elif selectedmode == 'Single droplet':
        # 保留单液滴模式的行，如果有
        results = results[results['Variable name'].isin(model_single)]
    data = results.values.tolist()
    video_name = video_name.split('/')[-2] + '/' + video_name.split('/')[-1]
    # 构建视频的访问URL
    video_path = url_for('get_result_file', filename=video_name, _external=True)
    print(video_path)
    # 返回两个处理后的图片路径和表格数据
    return jsonify({'video_path': video_path, 'withcount': withcount,
                    'withoutcount': withoutcount, 'image_names': image_names}), 200


@app.route('/api/video_statistics', methods=['POST'])
def video_statistics():
    # 从请求中获取前端传递的参数
    data = request.get_json()
    actual_size = data.get('actualSize')
    unit = data.get('unit')
    timeSpan = data.get('timeSpan')
    flowDirection = data.get('flowDirection')
    referenceLine = data.get('referenceLine')
    selectedmode = data.get('selectedMode')
    selectedalgorithm = data.get('selectedAlgorithm')
    print('输入参数')
    print('actual_size:', actual_size)
    print('unit:', unit)
    print('timeSpan:', timeSpan)
    print('flowDirection:', flowDirection)
    print('referenceLine:', referenceLine)
    print('selectedMode:', selectedmode)
    # 假设已经上传并保存了图像文件
    filename = data.get('filename')
    print(filename)
    # 调用图像处理函数，仅返回统计数据
    results, _ ,image_names= video_recognition(filename, app.config['RESULT_FOLDER'] + '/', actual_size=actual_size,
                                               actual_unit = unit
                                   , time_span=timeSpan, flow_direction=flowDirection, reference_line=referenceLine,
                                      statistical_mode=selectedmode,
                                        algorithm=selectedalgorithm)
    results = results.fillna('null')
    withcount, withoutcount = getwithcount_video(results, unit, selectedmode)
    withcount = round(withcount, 2)
    withoutcount = round(withoutcount, 2)
    # 去除Double_single_count、Double_count、Single_count行
    results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
    # 根据模式保留对应的行
    mode_double = ['core_num','shell_diameter_'+unit,'core_diameter_'+unit,'single_diameter_'+unit,
                   'concentricity','core_shell_ratio','volume_ratio']
    model_single = ['single_diameter_'+unit]
    if selectedmode == 'Double emulsion' or selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
        # 保留双液滴模式的行，如果有
        results = results[results['Variable name'].isin(mode_double)]
    elif selectedmode == 'Single droplet':
        # 保留单液滴模式的行，如果有
        results = results[results['Variable name'].isin(model_single)]
    # 确保四舍五入操作仅对数值类型的列进行
    numeric_columns = results.select_dtypes(include=['float64', 'int64']).columns
    results[numeric_columns] = results[numeric_columns].round(2)

    data = results.values.tolist()
    columns = results.columns.tolist()
    # 合并列名和数据
    stata_data = [columns] + data
    # 返回统计数据
    return jsonify({'statistics': stata_data, 'withcount': withcount, 'withoutcount': withoutcount}), 200


@app.route('/api/img_visualize', methods=['POST'])
def img_visualize():
    data = request.get_json()
    actual_size = data.get('actualSize')
    unit = data.get('unit')
    selectedmode = data.get('selectedMode')
    selectedalogrithm = data.get('selectedAlgorithm')
    print('输入参数')
    print('actual_size:', actual_size)
    print('unit:', unit)
    print('selectedmode:', selectedmode)
    print('seclectedAlgorithm:', selectedalogrithm)

    # 处理图像并获取结果路径和表格数据
    filename = data.get('filename')
    # 获取没有后缀的文件名
    results, image_names = image_recognition(filename, app.config['RESULT_FOLDER'] + '/',
                                             actual_size = actual_size, actual_unit = unit, statistical_mode=selectedmode,
                                                algorithm=selectedalogrithm)
    # 将results中的Nan替换为null
    results = results.fillna('null')
    withcount, withoutcount = getwithcount(results, unit, selectedmode)
    # 去除Double_single_count、Double_count、Single_count行
    results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
    # 根据模式保留对应的行
    mode_double = ['core_num','shell_diameter_'+unit,'core_diameter_'+unit,'single_diameter_'+unit,
                   'concentricity','core_shell_ratio','volume_ratio']
    model_single = ['single_diameter_'+unit]
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
    print('输入参数')
    print('actual_size:', actual_size)
    print('unit:', unit)
    print('selectedMode:', selectedmode)
    print('seclectedAlgorithm:', selectedAlgorithm)
    # 假设已经上传并保存了图像文件
    filename = data.get('filename')
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # 调用图像处理函数，仅返回统计数据
    results, _ = image_recognition(filename, app.config['RESULT_FOLDER'] + '/', actual_size=actual_size, actual_unit=unit,
                                   statistical_mode=selectedmode, algorithm=selectedAlgorithm)
    # 将results中的Nan替换为null
    results = results.fillna('null')
    withcount, withoutcount = getwithcount(results, unit, selectedmode)
    # 去除Double_single_count、Double_count、Single_count行
    results = results[~results['Variable name'].isin(['Double_single_count', 'Double_count', 'Single_count'])]
    # 根据模式保留对应的行
    mode_double = ['core_num','shell_diameter_'+unit,'core_diameter_'+unit,'single_diameter_'+unit,
                   'concentricity','core_to_shell_ratio','volume_ratio']
    model_single = ['single_diameter_'+unit]
    if selectedmode == 'Double emulsion' or selectedmode == 'Cell encapsulation' or selectedmode == 'Single-cell encapsulated':
        # 保留双液滴模式的行，如果有
        results = results[results['Variable name'].isin(mode_double)]
    elif selectedmode == 'Single droplet':
        # 保留单液滴模式的行，如果有
        results = results[results['Variable name'].isin(model_single)]

    data = results.values.tolist()
    columns = results.columns.tolist()
    # 合并列名和数据
    stata_data = [columns] + data
    print(stata_data)
    # 返回统计数据
    return jsonify({'statistics': stata_data, 'withcount': withcount, 'withoutcount': withoutcount}), 200




if __name__ == '__main__':
    # app.run(debug=True)
    app.run()