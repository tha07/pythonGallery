"""
Expose tflite models via a rest API.
"""
import io
import sys

import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from helpers import classify_image, read_labels, set_input_tensor

app = FastAPI()

# Settings
MIN_CONFIDENCE = 0.1  # The absolute lowest confidence for a detection.
# URL
FACE_DETECTION_URL = "/v1/vision/face"
OBJ_DETECTION_URL = "/v1/vision/detection"
SCENE_URL = "/v1/vision/scene"
IMAGE_URL = "/v1/vision/image"
# Models and labels
FACE_MODEL = "models/face_detection/mobilenet_ssd_v2_face/mobilenet_ssd_v2_face_quant_postprocess.tflite"
OBJ_MODEL = "models/object_detection/mobilenet_ssd_v2_coco/mobilenet_ssd_v2_coco_quant_postprocess.tflite"
OBJ_LABELS = "models/object_detection/mobilenet_ssd_v2_coco/coco_labels.txt"
SCENE_MODEL = "models/classification/dogs-vs-cats/model.tflite"
SCENE_LABELS = "models/classification/dogs-vs-cats/labels.txt"
IMAGE_MODEL = "models/image_class/mobilenet_v1_1.0_224_quant.tflite"
IMAGE_LABELS = "models/image_class/labelsnumbered.txt"

# Setup object detection
obj_interpreter = tflite.Interpreter(model_path=OBJ_MODEL)
obj_interpreter.allocate_tensors()
obj_input_details = obj_interpreter.get_input_details()
obj_output_details = obj_interpreter.get_output_details()
obj_input_height = obj_input_details[0]["shape"][1]
obj_input_width = obj_input_details[0]["shape"][2]
obj_labels = read_labels(OBJ_LABELS)

# Setup face detection
face_interpreter = tflite.Interpreter(model_path=FACE_MODEL)
face_interpreter.allocate_tensors()
face_input_details = face_interpreter.get_input_details()
face_output_details = face_interpreter.get_output_details()
face_input_height = face_input_details[0]["shape"][1]
face_input_width = face_input_details[0]["shape"][2]

# Setup face detection
scene_interpreter = tflite.Interpreter(model_path=SCENE_MODEL)
scene_interpreter.allocate_tensors()
scene_input_details = scene_interpreter.get_input_details()
scene_output_details = scene_interpreter.get_output_details()
scene_input_height = scene_input_details[0]["shape"][1]
scene_input_width = scene_input_details[0]["shape"][2]
scene_labels = read_labels(SCENE_LABELS)


# Setup image detection
image_interpreter = tflite.Interpreter(model_path=IMAGE_MODEL)
image_interpreter.allocate_tensors()
image_input_details = image_interpreter.get_input_details()
image_output_details = image_interpreter.get_output_details()

floating_model = image_input_details[0]['dtype'] == np.float32

image_input_height = image_input_details[0]["shape"][1]
image_input_width = image_input_details[0]["shape"][2]

image_labels = read_labels(IMAGE_LABELS)


@app.get("/")
async def info():
    return """tflite-server docs at ip:port/docs"""


@app.post(FACE_DETECTION_URL)
async def predict_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((face_input_width, face_input_height), Image.ANTIALIAS)
        input_data = np.expand_dims(resized_image, axis=0)
        face_interpreter.set_tensor(face_input_details[0]["index"], input_data)

        # Process image and get predictions
        face_interpreter.invoke()
        boxes = face_interpreter.get_tensor(face_output_details[0]["index"])[0]
        classes = face_interpreter.get_tensor(face_output_details[1]["index"])[0]
        scores = face_interpreter.get_tensor(face_output_details[2]["index"])[0]

        data = {}
        faces = []
        for i in range(len(scores)):
            if not classes[i] == 0:  # Face
                continue
            single_face = {}
            single_face["userid"] = "unknown"
            single_face["confidence"] = float(scores[i])
            single_face["y_min"] = int(float(boxes[i][0]) * image_height)
            single_face["x_min"] = int(float(boxes[i][1]) * image_width)
            single_face["y_max"] = int(float(boxes[i][2]) * image_height)
            single_face["x_max"] = int(float(boxes[i][3]) * image_width)
            if single_face["confidence"] < MIN_CONFIDENCE:
                continue
            faces.append(single_face)

        data["predictions"] = faces
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post(OBJ_DETECTION_URL)
async def predict_object(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((obj_input_width, obj_input_height), Image.ANTIALIAS)
        input_data = np.expand_dims(resized_image, axis=0)
        obj_interpreter.set_tensor(obj_input_details[0]["index"], input_data)

        # Process image and get predictions
        obj_interpreter.invoke()
        boxes = obj_interpreter.get_tensor(obj_output_details[0]["index"])[0]
        classes = obj_interpreter.get_tensor(obj_output_details[1]["index"])[0]
        scores = obj_interpreter.get_tensor(obj_output_details[2]["index"])[0]

        data = {}
        objects = []
        for i in range(len(scores)):
            single_object = {}
            single_object["label"] = obj_labels[int(classes[i])]
            single_object["confidence"] = float(scores[i])
            single_object["y_min"] = int(float(boxes[i][0]) * image_height)
            single_object["x_min"] = int(float(boxes[i][1]) * image_width)
            single_object["y_max"] = int(float(boxes[i][2]) * image_height)
            single_object["x_max"] = int(float(boxes[i][3]) * image_width)

            if single_object["confidence"] < MIN_CONFIDENCE:
                continue
            objects.append(single_object)

        data["predictions"] = objects
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post(SCENE_URL)
async def predict_scene(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        resized_image = image.resize((scene_input_width, scene_input_height), Image.ANTIALIAS)
        results = classify_image(scene_interpreter, image=resized_image)
        label_id, prob = results[0]
        data = {}
        data["label"] = scene_labels[label_id]
        data["confidence"] = prob
        data["success"] = True
        return data
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


@app.post(IMAGE_URL)
async def predict_object(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image = Image.open(io.BytesIO(contents))
        image_width = image.size[0]
        image_height = image.size[1]

        # Format data and send to interpreter
        resized_image = image.resize((image_input_width, image_input_height), Image.ANTIALIAS)
        input_data = np.expand_dims(resized_image, axis=0)
        image_interpreter.set_tensor(image_input_details[0]["index"], input_data)


        # Process image and get predictions
        image_interpreter.invoke()
        output_data = image_interpreter.get_tensor(image_output_details[0]['index'])

        #boxes = image_interpreter.get_tensor(image_output_details[0]["index"])[0]
        #classes = image_interpreter.get_tensor(image_output_details[1]["index"])[0]
        #scores = image_interpreter.get_tensor(image_output_details[2]["index"])[0]

        results = np.squeeze(output_data)

        top_k = results.argsort()[-5:][::-1]

        data = {}
        objects = []

        single_object = {}
        single_object["label"] = image_labels[top_k[0]]
        single_object["label"] = single_object["label"].split(",")
        single_object["label"] = single_object["label"][0]
        single_object["confidence"] = float(results[top_k[0]])/255.0

        objects.append(single_object)
        data["predictions"] = objects
        data["success"] = True

        return data


    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))
