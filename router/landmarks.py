from fastapi import APIRouter, File, UploadFile
from controllers import landmarksController


router = APIRouter()


@router.post('/api/landmarks', response_model=str)
async def landmarks(file: UploadFile = File(...)):
    return await landmarksController.detect_landmarks(file)
