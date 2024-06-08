import onnxruntime as ort
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import UploadFile
from utils import read_image, preprocess_image, get_landmarks
import numpy as np


async def detect_landmarks(file: UploadFile):
    labels = [
        "left_eye_center",
        "right_eye_center",
        "left_eye_inner_corner",
        "left_eye_outer_corner",
        "right_eye_inner_corner",
        "right_eye_outer_corner",
        "left_eyebrow_inner_end",
        "left_eyebrow_outer_end",
        "right_eyebrow_inner_end",
        "right_eyebrow_outer_end",
        "nose_tip",
        "mouth_left_corner",
        "mouth_right_corner",
        "mouth_center_top_lip",
        "mouth_center_bottom_lip"
    ]

    image_size = (320, 320)
    model = ort.InferenceSession("model/face-landmarks.onnx")
    origin_image = await read_image(file)
    preprocessed_image = preprocess_image(origin_image, image_size)

    outputs = model.run(None, {'images': [preprocessed_image]})
    output = np.transpose(outputs[0], (0, 2, 1))
    output = np.squeeze(output)

    max_conf_idx = np.argmax(output[:, 4])
    print('landmarks', output[max_conf_idx])

    if output[max_conf_idx, 4] < 0.6:
        JSONResponse(content={"landmarks": {}})
        # raise HTTPException(status_code=400, detail="Could not reliably detect landmarks.")

    landmarks = output[max_conf_idx, 5:]
    landmarks = get_landmarks(origin_image, landmarks, labels)
    response_data = {
        "landmarks": landmarks
    }

    return JSONResponse(content=response_data)
