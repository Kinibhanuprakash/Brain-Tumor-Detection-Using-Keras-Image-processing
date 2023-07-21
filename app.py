import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
from flask import Flask, render_template, request
from keras.preprocessing import image
from tensorflow import keras
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename

global graph
sess = tf.Session()
graph=tf.get_default_graph()

# Define a flask app
app = Flask(__name__)
set_session(sess)
# Load your trained model
model = load_model('braintumor1.h5',compile=False)




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('brainintro.html')

@app.route('/predict', methods=['GET'])
def predict1():
    # Main page
    return render_template('brainpred2.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        df= pd.read_csv('patient.csv')
        f = request.files['image']
        name=request.form['name']
        age=request.form['age']
        
        # Saves the images to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img = image.load_img(file_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        with graph.as_default():
            model = load_model('braintumor1.h5',compile=False)
            prediction=np.argmax(model.predict(x), axis=1)
        print(prediction)
        if prediction==0:
            text = "You are perfectly fine"
            inp = "No tumor"
        else:
            text = "You are infected! Please Consult Doctor"
            inp="Tumor detected"
        df=df._append(pd.DataFrame({'name':[name],'age':[age],'status':[inp]}),ignore_index=True)
        df.to_csv('patient.csv',index = False)
        return text
if __name__ == '__main__':
    app.run(debug=True,port=144)
