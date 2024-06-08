from fastapi import APIRouter, File, UploadFile
from controllers import comicController


router = APIRouter()


@router.post('/api/comic', response_model=str)
async def comic(file: UploadFile = File(...)):
    return await comicController.comic(file)
