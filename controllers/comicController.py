import onnxruntime as ort
from fastapi.responses import StreamingResponse
from fastapi import UploadFile
from utils import *
from io import BytesIO
from PIL import Image
import numpy as np


async def comic(file: UploadFile):
    model = ort.InferenceSession('model/face2comic.onnx')
    input_info = model.get_inputs()[0]

    input_name = input_info.name
    input_shape = input_info.shape
    img_h, img_w = input_shape[-2:]
    output_name = model.get_outputs()[0].name

    origin_image = await read_image(file)
    preprocessed_image = preprocess_image(origin_image, (img_w, img_h))
    preprocessed_image = denormalize(preprocessed_image)

    outputs = model.run([output_name], {input_name: [preprocessed_image]})
    output_image = outputs[0].squeeze().transpose(1, 2, 0)

    output_image = (normalize(output_image) * 255).astype(np.uint8)
    output_image = Image.fromarray(output_image)

    buf = BytesIO()
    output_image.save(buf, format='PNG')
    buf.seek(0)

    return StreamingResponse(buf, media_type='image/png')
