from fastapi import APIRouter, File, UploadFile
from controllers import genAttributesController


router = APIRouter()


@router.post('/api/gen-attributes', response_model=str)
async def gen_attributes(file: UploadFile = File(...)):
    return await genAttributesController.genAttributes(file)
