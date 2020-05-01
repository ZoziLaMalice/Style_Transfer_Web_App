from flask import Flask, render_template, request, url_for, flash, redirect
from flask import send_from_directory

import os
import IPython.display as display
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
from werkzeug.utils import secure_filename

from model_functions import normalize, get_model, gram_matrix, clip
from model_functions import StyleContentModel, style_content_loss, tensor_to_image



app = Flask(__name__)


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
OUTPUT_FOLDER = './static/output'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        global epoch
        epoch = request.form.get('epochs')
        global steps
        steps = request.form.get('steps')
        # check if the post request has the file part
        if 'upload_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['upload_file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('styled', filename=filename))

    return render_template('index.html')


@app.route('/uploads/<filename>')
def styled(filename):
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = ['block5_conv2']

    style_extractor = get_model(style_layers)
    content_extractor = get_model(content_layer)

    model = StyleContentModel(style_layers=style_layers, content_layer=content_layer)

    style_targets = model(normalize('./style/kandinsky.jpg'))['style']
    content_targets = model(normalize(UPLOAD_FOLDER+ "/"+filename))['content']
    
    image = tf.Variable(normalize(UPLOAD_FOLDER+ "/"+filename))
    
    
    EPOCHS = int(epoch)
    steps_per_epoch = int(steps)

    step = 0
    start = time.time()

    for n in range(EPOCHS):
        for m in range(steps_per_epoch):
            step += 1
            with tf.GradientTape() as tape:
                outputs = model(image)
                loss = style_content_loss(outputs, style_targets, content_targets, style_layers, content_layer, 1e-2, 1e4)

            grad = tape.gradient(loss, image)
            optimizer.apply_gradients([(grad, image)])
            image.assign(clip(image))
    
        print("Train step: {}".format(step))
    #model.save('my_model.h5')

    tensor_to_image(image).save(OUTPUT_FOLDER+ "/"+filename)

    end = time.time()
    print("Total time: {:.1f}".format(end-start))

    return render_template("output.html", user_image = filename)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'


    app.run(debug=True)
