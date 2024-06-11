from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from router import detect, landmarks, comic, attributes, genAttributes

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def home():
    return {'health_check': 'OK', 'message': 'Hello world!'}


@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "UP"}, status_code=200)


app.include_router(detect.router)
app.include_router(landmarks.router)
app.include_router(comic.router)
app.include_router(attributes.router)
app.include_router(genAttributes.router)


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
