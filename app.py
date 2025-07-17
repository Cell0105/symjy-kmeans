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
    count = len([name for name in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, name))])
    
    if request.method == 'POST':
        img = request.files['image']
        if img:
            filename = secure_filename(img.filename.replace(" ", "_"))
            filename = f"{uuid.uuid4().hex}_{filename}"
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            img.save(input_path)

            output_path, details, steps = segment_image(input_path, k=3)
            count += 1

            return render_template('SYMJY.html',
                       original=filename,
                       segmented=os.path.basename(output_path),
                       details=details,
                       steps=steps,
                       count=count)


    return render_template('SYMJY.html', count=count)



@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
   port = int(os.environ.get("PORT", 5000))
   app.run(host="0.0.0.0", port=port)
