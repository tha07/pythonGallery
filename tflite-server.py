################################ΔΗΛΩΣΕΙΣ ΒΙΒΛΙΟΘΗΚΩΝ#####################################################################
import io
import sys
import time
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

# Κάθε μοντέλο μηχανικής μάθησης φορτώνεται απο την βιβλιοθήκη του tensorflow.keras και στη συνέχεια είναι απαραίτητες οι συναρτήσεις
# preprocess_input και decode_predictions για την προεπεξεργασία των εικόνων και την αποκωδικοποίηση των αποτελεσμάτων αντίστοιχα για κάθε μοντέλο μηχανικής μάθησης 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocessResnet50
from tensorflow.keras.applications.resnet50 import decode_predictions as decodeResnet50
import numpy as np


from tensorflow.python.keras.applications.efficientnet import EfficientNetB0, EfficientNetB7
from tensorflow.python.keras.applications.efficientnet import preprocess_input as preprocessEfficientB0
from tensorflow.python.keras.applications.efficientnet import decode_predictions as decodeEfficientB0

from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.applications.nasnet import preprocess_input as preprocessNasLarge
from tensorflow.python.keras.applications.nasnet import decode_predictions as decodeNasLarge

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input as preprocessVGG16
from tensorflow.python.keras.applications.vgg16 import decode_predictions as decodeVGG16

from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_resnet_v2 import preprocess_input as preprocessInRes
from tensorflow.python.keras.applications.inception_resnet_v2 import decode_predictions as decodeInRes

from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.applications.xception import preprocess_input as preprocessXception
from tensorflow.python.keras.applications.xception import  decode_predictions as decodeXception

from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.inception_v3 import preprocess_input as preprocesInception
from tensorflow.python.keras.applications.inception_v3 import decode_predictions as decodeInception


from helpers import classify_image, read_labels, set_input_tensor
import tensorflow as tf
###############################################################################################################3

#Δήλωση 
app = FastAPI()

#Δήλωση μοντέλων μηχανικής μάθησης και οι αντίστοιχες ρυθμίσεις τους
modelRes = ResNet50(weights='imagenet')
modelVGG = VGG16(weights='imagenet', include_top=True)
modelEfficient = EfficientNetB0(include_top=True, weights='imagenet')
modelEfficientB7 = EfficientNetB7(include_top=True, weights='imagenet')
modelNasNet = NASNetLarge(include_top=True, weights='imagenet')
modelMobileNet = MobileNetV2(weights="imagenet", include_top=True)
modelXception = Xception(include_top=True,weights="imagenet")
modelInception = InceptionV3(include_top=True, weights="imagenet", classifier_activation="softmax")
modelInRes = InceptionResNetV2(weights="imagenet", include_top=True)

# Settings
MIN_CONFIDENCE = 0.1  # Το ελάχιστο δυνατό confidence που δέχόμαστε απο τα μοντέλα.

# Τα URLs που θα απαντάει το κάθε μοντέλο
IMAGE_URL2 = "/v1/vision/resNet"
IMAGE_URL3 = "/v1/vision/vggNet"
IMAGE_URL4 = "/v1/vision/efficientNet"
IMAGE_URL5 = "/v1/vision/nasNet"
IMAGE_URL6 = "/v1/vision/mobileNet2"
IMAGE_URL7 = "/v1/vision/xceptionNet"
IMAGE_URL8 = "/v1/vision/efficientNetB7"

# Ξεκινώντας στο domain "/" δηλαδή στην αρχική σελίδα του host του server εμφανίζεται το μήνυμα του return
@app.get("/")
async def info():
    return """tflite-server docs at ip:port/docs"""


# Στη συνέχεια το κάθε μοντέλο λειτουργεί με τις παρακάτω συναρτήσεις καλώντας το αντίστοιχο url
@app.post(IMAGE_URL2)
async def predict_scene(image: List[UploadFile] = File(...)): # Ως μεταβλητή, το image δέχεται λίστα με μία ή πολλαπλές εικόνες απο το post request της android εφαρμογής
    try:
        print("Welcome...") # Ένα μήνυμα καλωσορίσματος
        start = time.time()
        all_data = []
        all_predictions = []
        data = {}
        objects = []

        single_object = {}
        for i in tf.range(len(image)): # Για όλες τις εικόνες που βρίσκονται στη λίστα
            contents = await image[i].read() # Αναμένουμε το διάβασμα των περιεχομένων με το await ώστε να μην προχωρήσει χωρίς να διαβάσει αρχείο 
            image2 = Image.open(io.BytesIO(contents)) # Ανοίγουμε τα αρχεία εικόνας με τη βοήθεια της βιβλιοθήκης PIL 
            resized_image = image2.resize((224, 224), Image.ANTIALIAS) # Επαναλαμβάνουμε το resize για μικρή διόρθωση antialising
            input_data = np.expand_dims(resized_image, axis=0) # Δημιουργία πίνακα
            x = preprocessResnet50(input_data) # Επεξεργασία πινάκων ώστε να επεξεργαστούν σωστά απο το αντίστοιχο μοντέλο μηχανικής μάθησης
            all_data.append(x) # Προσθήκη όλων των προ επεξεργασμένων πινάκων κάθε εικόνας σε έναν

        reading_end = time.time()
        print("reading data.... " + (str(reading_end - start)))

        inference_time_start = time.time()
        dedomena = np.vstack(all_data) # Δημιουργία ενός πίνακα με όλους τους επεξεργασμένους πίνακες εικόνων σε έναν καθέτως - ανα σειρά
        megethos = len(all_data) # εύρεση μεγέθους πινακα
        preds = modelRes.predict(dedomena, batch_size=megethos) # Είσοδος δεδομένων εικόνων στο inference του μοντέλου επιλογής 
        predictions = decodeResnet50(preds, top=1) # Αποκωδικοποίηση των αποτελεσμάτων του inference βάσει του αντίστοιχου μοντέλου

        inference_time_end = time.time()
        print(inference_time_end - inference_time_start)

        labels = [] 
        time_spent = []
        time_spent.append(reading_end - start)
        time_spent.append(inference_time_end - inference_time_start)
		
        print(predictions[i][0][1])
        for i in range(len(all_data)): # Προσθήκη αποτελεσμάτων ταξινόμησης εικόνων σε ένα πίνακα
            labels.append(predictions[i][0][1])
        data["labels"] = labels # Προσθήκη labels σε λεξικό 
        data["times"] = time_spent # Προσθήκη χρόνων διαδικασιών στο λεξικό 
        return data # Επιστροφή λεξικού ως response στο post request 
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e)) # Σε περίπτωση λάθους επιστροφή error 500

# Κάθε μοντέλο στην συνέχεια, αποτελείται απο τα ίδια βήματα 

@app.post(IMAGE_URL3)
async def predict_scene(image: List[UploadFile] = File(...)):
    try:
        print("Welcome...")
        start = time.time()
        all_data = []
        all_predictions = []
        data = {}
        objects = []

        single_object = {}
        for i in tf.range(len(image)):
            contents = await image[i].read()
            image2 = Image.open(io.BytesIO(contents))
            resized_image = image2.resize((224, 224), Image.ANTIALIAS)
            input_data = np.expand_dims(resized_image, axis=0)
            x = preprocessVGG16(input_data)
            all_data.append(x)

        reading_end = time.time()
        print("reading data.... " + (str(reading_end - start)))

        inference_time_start = time.time()
        dedomena = np.vstack(all_data)
        megethos = len(all_data)
        preds = modelVGG.predict(dedomena, batch_size=megethos)

        predictions = decodeVGG16(preds, top=1)

        inference_time_end = time.time()
        print(inference_time_end - inference_time_start)

        labels = []
        time_spent = []
        time_spent.append(reading_end - start)
        time_spent.append(inference_time_end - inference_time_start)

        print(predictions[i][0][1])
        for i in range(len(all_data)):
            labels.append(predictions[i][0][1])
        data["labels"] = labels
        data["times"] = time_spent

        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post(IMAGE_URL4)
async def predict_scene(image: List[UploadFile] = File(...)):
    try:
        print("Welcome...")
        start = time.time()
        all_data = []
        all_predictions = []
        data = {}
        objects = []

        single_object = {}
        for i in tf.range(len(image)):
            contents = await image[i].read()
            image2 = Image.open(io.BytesIO(contents))
            resized_image = image2.resize((224, 224), Image.ANTIALIAS)
            input_data = np.expand_dims(resized_image, axis=0)
            x = preprocessEfficientB0(input_data)
            all_data.append(x)

        reading_end = time.time()
        print("reading data.... " + (str(reading_end - start)))


        inference_time_start = time.time()


        dedomena = np.vstack(all_data)
        megethos = len(all_data)
        preds = modelEfficient.predict(dedomena, batch_size=megethos)
        # for i in range(len(all_data)):

        predictions = decodeEfficientB0(preds, top=1)

        inference_time_end = time.time()
        print(inference_time_end - inference_time_start)

        labels = []
        time_spent = []
        time_spent.append(reading_end - start)
        time_spent.append(inference_time_end - inference_time_start)

        print(predictions[i][0][1])
        for i in range(len(all_data)):
            labels.append(predictions[i][0][1])
        data["labels"] = labels
        data["times"] = time_spent

        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post(IMAGE_URL5)
async def predict_scene(image: List[UploadFile] = File(...)):
    try:
        print("Welcome...")
        start = time.time()
        all_data = []
        all_predictions = []
        data = {}
        objects = []

        single_object = {}
        for i in tf.range(len(image)):
            contents = await image[i].read()
            image2 = Image.open(io.BytesIO(contents))
            resized_image = image2.resize((331, 331), Image.ANTIALIAS)
            input_data = np.expand_dims(resized_image, axis=0)
            x = preprocessNasLarge(input_data)
            all_data.append(x)

        reading_end = time.time()
        print("reading data.... " + (str(reading_end - start)))


        inference_time_start = time.time()
        dedomena = np.vstack(all_data)
        megethos = len(all_data)
        preds = modelNasNet.predict(dedomena, batch_size=megethos)


        predictions = decodeNasLarge(preds, top=1)

        inference_time_end = time.time()
        print(inference_time_end - inference_time_start)

        labels = []
        time_spent = []
        time_spent.append(reading_end - start)
        time_spent.append(inference_time_end - inference_time_start)

        print(predictions[i][0][1])
        for i in range(len(all_data)):
            labels.append(predictions[i][0][1])
        data["labels"] = labels
        data["times"] = time_spent

        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post(IMAGE_URL6)
async def predict_scene(image: List[UploadFile] = File(...)):
    try:
        print("Welcome...")
        start = time.time()
        all_data=[]
        all_predictions = []
        data = {}
        objects = []

        single_object = {}
        for i in tf.range(len(image)):
            contents = await image[i].read()
            image2 = Image.open(io.BytesIO(contents))
            resized_image = image2.resize((299, 299), Image.ANTIALIAS)
            input_data = np.expand_dims(resized_image, axis=0)
            x = preprocessInRes(input_data)
            all_data.append(x)

        reading_end = time.time()
        print("reading data.... "+ (str(reading_end- start )))
        #print(len(all_data))

        inference_time_start = time.time()
        dedomena = np.vstack(all_data)
        megethos = len(all_data)
        preds = modelInRes.predict(dedomena, batch_size=megethos)
        #for i in range(len(all_data)):

        predictions = decodeInRes(preds, top=1)

        inference_time_end = time.time()
        print(inference_time_end-inference_time_start)
        #print(len(data['predictions']))
        #return data



        labels = []
        time_spent = []
        time_spent.append(reading_end-start)
        time_spent.append(inference_time_end-inference_time_start)

        print(predictions[i][0][1])
        for i in range(len(all_data)):
            labels.append(predictions[i][0][1])
        data["labels"] = labels
        data["times"] = time_spent


        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post(IMAGE_URL7)
async def predict_scene(image: List[UploadFile] = File(...)):
    try:
        print("Welcome...")
        start = time.time()
        all_data = []
        all_predictions = []
        data = {}
        objects = []

        single_object = {}

        for i in tf.range(len(image)):
            contents = await image[i].read()
            image2 = Image.open(io.BytesIO(contents))
            resized_image = image2.resize((299, 299), Image.ANTIALIAS)
            input_data = np.expand_dims(resized_image, axis=0)
            x = preprocessXception(input_data)
            all_data.append(x)

        reading_end = time.time()
        print("reading data.... " + (str(reading_end - start)))


        inference_time_start = time.time()
        dedomena = np.vstack(all_data)
        megethos = len(all_data)
        preds = modelXception.predict(dedomena, batch_size= megethos)
        # for i in range(len(all_data)):

        predictions = decodeXception(preds, top=1)

        inference_time_end = time.time()
        print(inference_time_end - inference_time_start)

        labels = []
        time_spent = []
        time_spent.append(reading_end - start)
        time_spent.append(inference_time_end - inference_time_start)

        print(predictions[i][0][1])
        for i in range(len(all_data)):
            labels.append(predictions[i][0][1])
        data["labels"] = labels
        data["times"] = time_spent

        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))

@app.post(IMAGE_URL8)
async def predict_scene(image: List[UploadFile] = File(...)):
    try:
        print("Welcome...")
        start = time.time()
        all_data = []
        all_predictions = []
        data = {}
        objects = []

        single_object = {}

        for i in tf.range(len(image)):
            contents = await image[i].read()
            image2 = Image.open(io.BytesIO(contents))
            resized_image = image2.resize((600, 600), Image.ANTIALIAS)
            input_data = np.expand_dims(resized_image, axis=0)
            x = preprocessEfficientB0(input_data)
            all_data.append(x)

        reading_end = time.time()
        print("reading data.... " + (str(reading_end - start)))
        # print(len(all_data))


        inference_time_start = time.time()
        dedomena = np.vstack(all_data)
        megethos = len(all_data)
        preds = modelEfficientB7.predict(dedomena, batch_size= megethos)

        predictions = decodeEfficientB0(preds, top=1)

        inference_time_end = time.time()
        print(inference_time_end - inference_time_start)

        labels = []
        time_spent = []
        time_spent.append(reading_end - start)
        time_spent.append(inference_time_end - inference_time_start)

        print(predictions[i][0][1])
        for i in range(len(all_data)):
            labels.append(predictions[i][0][1])
        data["labels"] = labels
        data["times"] = time_spent

        return data

    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
