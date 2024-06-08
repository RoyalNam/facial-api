from fastapi import APIRouter, File, UploadFile
from controllers import detectController

router = APIRouter()


@router.post('/api/detect', response_model=str)
async def detect(file: UploadFile = File(...)):
    return await detectController.detect(file)
