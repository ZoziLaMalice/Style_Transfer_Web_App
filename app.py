from flask import Flask, render_template, request, url_for, flash, redirect
from flask import send_from_directory

import os
import IPython.display as display
import PIL
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
from werkzeug.utils import secure_filename

from model_functions import normalize, vgg_layers, gram_matrix, clip_0_1, train_step, train_step_bis
from model_functions import StyleContentModel, style_content_loss, tensor_to_image

tf.config.experimental_run_functions_eagerly(True)

app = Flask(__name__)


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
OUTPUT_FOLDER = './static/output'
STYLE_FOLDER = './static/style'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['STYLE_FOLDER'] = STYLE_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':

        global epoch
        epoch = request.form.get('epochs')
        if epoch == "":
            epoch = 10
        
        global steps
        steps = request.form.get('steps')
        if steps == "":
            steps = 10

        global img_style
        img_style = request.form.get('style')

        global style_method
        style_method = request.form.get('btn')

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
    if style_method == "slow":
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        content_layer = ['block5_conv2']

        style_extractor = vgg_layers(style_layers)
        content_extractor = vgg_layers(content_layer)

        extractor = StyleContentModel(style_layers=style_layers, content_layers=content_layer)

        style_targets = extractor(normalize(STYLE_FOLDER + "/" + img_style))['style']
        content_targets = extractor(normalize(UPLOAD_FOLDER + "/" + filename))['content']
        
        image = tf.Variable(normalize(UPLOAD_FOLDER + "/" + filename))

        start = time.time()

        EPOCHS = int(epoch)
        steps_per_epoch = int(steps)

        step = 0
        for n in range(EPOCHS):
            for m in range(steps_per_epoch):
                step += 1
                #train_step(image, style_targets, content_targets, style_layers, content_layer, style_weight=1e-2, content_weight=1e-4)
                train_step_bis(image, style_targets, content_targets, style_layers, content_layer, style_weight=1e-2, content_weight=1e-4, total_variation_weight=30)

            print("Train step: {}".format(step))
          
        end = time.time()
        print("Total time: {:.1f}".format(end-start))

        tensor_to_image(image).save(OUTPUT_FOLDER + "/" + filename)

    else:
        hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        style_image = normalize(STYLE_FOLDER + "/" + img_style)
        content_image = normalize(UPLOAD_FOLDER + "/" + filename)

        stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
        tensor_to_image(stylized_image).save(OUTPUT_FOLDER + "/" + filename)

    return render_template("output.html", user_image = filename)


if __name__ == "__main__":
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'


    app.run(debug=True)
