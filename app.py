from flask import Flask, render_template, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
app = Flask(__name__, static_folder='main_static_dir')

@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/images/<path:filename>")
def base_static(filename):
    return send_from_directory(app.root_path + '/images', filename)

UPLOAD_FOLDER = os.getcwd() + '/images/uploaded'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

from detector import detect
@app.route('/', methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            detect(filename)
            return render_template('home.html', picture=filename)

    return render_template('home.html', picture="../../images/static/2.png")

if __name__ == '__main__':
    app.run(debug=True)