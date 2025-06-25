from flask import Flask, render_template, request, send_file
from kmeans_segment import segment_image
from werkzeug.utils import secure_filename
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img = request.files['image']
        if img:
            filename = secure_filename(img.filename.replace(" ", "_"))
            filename = f"{uuid.uuid4().hex}_{filename}"
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            img.save(input_path)

            output_path, details = segment_image(input_path, k=3)

            return render_template('SYMJY.html',
                       original=filename,
                       segmented=os.path.basename(output_path),
                       details=details)

    return render_template('SYMJY.html')

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
