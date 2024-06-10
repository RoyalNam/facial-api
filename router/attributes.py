from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from controllers.attributesController import FaceAttributes


router = APIRouter()
faceAttributes = FaceAttributes(
    'model/face-age.onnx',
    'model/face-gender.onnx',
    'model/face-emotion.onnx',
)


@router.post('/api/age', response_model=str)
async def age(file: UploadFile = File(...)):
    age_data = await faceAttributes.age(file)
    return JSONResponse(age_data)


@router.post('/api/gender', response_model=str)
async def gender(file: UploadFile = File(...)):
    gender_data = await faceAttributes.gender(file)
    return JSONResponse(gender_data)


@router.post('/api/emotion', response_model=str)
async def emotion(file: UploadFile = File(...)):
    emotion_data = await faceAttributes.emotion(file)
    return JSONResponse(emotion_data)


@router.post('/api/attributes', response_model=str)
async def attributes(file: UploadFile = File(...)):
    attrs_data = await faceAttributes.analyze_face_attributes(file)
    return JSONResponse(attrs_data)
