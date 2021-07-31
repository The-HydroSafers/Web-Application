from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from tensorflow.keras.models import load_model,Model,Sequential
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
model=load_model("hydro.h5")
app=Flask(__name__)
@app.route('/')
def index():
    # Main page
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    request_data=request.get_json(force=True)
    img=request_data['image']
    img=np.array(img).reshape((-1,224,224,3))
    return ("The prediction is{}".format(['Chlorophyll','Milk','DeepBlue'][model.predict_classes(img).argmax()]))

if __name__=='__main__':
    app.run(host='127.0.0.1',port=8080)