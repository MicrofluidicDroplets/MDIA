from flask import Flask, jsonify, request, render_template

app = Flask(__name__, static_folder='static', template_folder='templates')

# 简单的文章数据存储
posts = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/posts', methods=['GET'])
def get_posts():
    return jsonify(posts)

@app.route('/api/posts', methods=['POST'])
def add_post():
    post = request.json
    posts.append(post)
    return jsonify(post), 201

if __name__ == '__main__':
    app.run(debug=True)
