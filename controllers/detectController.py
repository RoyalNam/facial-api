import onnxruntime as ort
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import UploadFile
from io import BytesIO
from utils import read_image, preprocess_image, NMSBoxes, get_boxes
import numpy as np
import json


async def detect(file: UploadFile):
    image_size = (640, 640)

    model = ort.InferenceSession("model/face-detect.onnx")
    original_image = await read_image(file)

    preprocessed_image = preprocess_image(original_image, image_size)
    outputs = model.run(None, {'images': [preprocessed_image]})
    output = np.transpose(outputs[0], (0, 2, 1))
    output = np.squeeze(output)
    boxes = output[:, :4]
    scores = output[:, 4]
    keep_indices = NMSBoxes(boxes, scores, confidence_threshold=0.45, nms_threshold=0.25)
    keep_indices_list = [box.tolist() for box in keep_indices]
    boxes = get_boxes(original_image, keep_indices_list, image_size)
    response_data = {
        "boxes": boxes
    }

    return JSONResponse(content=response_data)
