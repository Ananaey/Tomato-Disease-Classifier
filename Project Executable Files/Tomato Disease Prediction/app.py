from __future__ import division, print_function
import os

import flask
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

from flask_mail import Mail, Message

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Flask-Mail configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'ananaeygarg@gmail.com'
app.config['MAIL_PASSWORD'] = 'elsalko1#'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

MODEL_PATH = '.venv/Training/tmt.keras'
model = load_model("tmt.keras")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'uploads')  # Updated path to 'uploads'

        # Create the directory if it doesn't exist
        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        file_path = os.path.join(upload_path, secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        disease_steps = {
            "Bacterial_spot": "Remove infected plants. Avoid overhead watering. Apply appropriate bactericides.",
            "Early_blight": "Remove infected leaves. Apply fungicides. Rotate crops.",
            "Late_blight": "Remove infected plants. Apply fungicides. Ensure proper plant spacing.",
            "Leaf_Mold": "Improve air circulation. Apply fungicides. Avoid overhead watering.",
            "Septoria_leaf_spot": "Remove infected leaves. Apply fungicides. Rotate crops.",
            "Spider_mites Two-spotted_spider_mite": "Apply miticides. Introduce natural predators. Maintain proper plant hydration.",
            "Target_Spot": "Remove infected leaves. Apply fungicides. Rotate crops.",
            "Tomato_Yellow_Leaf_Curl_Virus": "Remove infected plants. Control whitefly population. Use virus-resistant varieties.",
            "Tomato_mosaic_virus": "Remove infected plants. Disinfect tools. Avoid handling plants when wet.",
            "Healthy": "Your plant is healthy. Keep up the good care!"
        }
        steps = disease_steps.get(preds, "No specific steps available.")
        image_url = f'uploads/{secure_filename(f.filename)}'  # Updated path to 'uploads'

        return redirect(url_for('result', disease=preds, steps=steps, image_url=image_url))
    return render_template('predict.html')


@app.route('/result', methods=['GET'])
def result():
    disease = request.args.get('disease')
    steps = request.args.get('steps')
    image_url = request.args.get('image_url')
    return render_template('result.html', disease=disease, steps=steps, image_url=image_url)


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        msg = Message(subject=subject,
                      sender=app.config['MAIL_USERNAME'],
                      recipients=['recipient@example.com'])
        msg.body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
        mail.send(msg)

        flash('Your message has been sent. Thank you!')
        return redirect('/contact')
    return render_template('contact.html')


@app.route('/uploads/<img_name>', methods=['GET'])
def uploaded_image(img_name):
    return flask.send_file(f'{app.root_path}/uploads/{img_name}', mimetype="image/jpg")


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    diseases = [
        "Bacterial_spot", "Early_blight", "Late_blight", "Leaf_Mold",
        "Septoria_leaf_spot", "Spider_mites Two-spotted_spider_mite",
        "Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Healthy"
    ]
    return diseases[preds[0]]


if __name__ == '__main__':
    app.run(port=5001, debug=True)
