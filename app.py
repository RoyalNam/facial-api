from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import onnxruntime as ort
from fastapi.responses import StreamingResponse
from utils import *


app = FastAPI()


@app.get("/")
def home():
    return {"health_check": "OK", "message": "Hello world!"}


@app.post('/api/detect', response_model=str)
async def detect(file: UploadFile = File(...)):
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

    image_pil = draw_box(original_image, keep_indices, image_size=image_size)
    image_byte_arr = BytesIO()
    image_pil.save(image_byte_arr, format='PNG')
    image_byte_arr.seek(0)
    return StreamingResponse(image_byte_arr, media_type='image/png')


@app.post('/api/landmarks', response_model=str)
async def keypoint(file: UploadFile = File(...)):
    image_size = (320, 320)
    model = ort.InferenceSession("model/face-landmarks.onnx")
    origin_image = await read_image(file)

    preprocessed_image = preprocess_image(origin_image, image_size)
    outputs = model.run(None, {'images': [preprocessed_image]})
    output = np.transpose(outputs[0], (0, 2, 1))
    output = np.squeeze(output)

    max_conf_idx = np.argmax(output[:, 4])
    print('landmarks', output[max_conf_idx])

    if output[max_conf_idx, 4] < 0.7:
        raise HTTPException(status_code=400, detail="Could not reliably detect landmarks.")

    landmarks = output[max_conf_idx, 5:]
    image = draw_landmarks(origin_image, landmarks, image_size=image_size)

    image_byte_arr = BytesIO()
    image.save(image_byte_arr, format='PNG')
    image_byte_arr.seek(0)
    return StreamingResponse(image_byte_arr, media_type='image/png')


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
