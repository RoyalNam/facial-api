import onnxruntime as ort
from fastapi import UploadFile
from utils import read_image, preprocess_image
from PIL import Image
import numpy as np


class FaceAttributes:
    def __init__(self, age_model_path, gender_model_path, emotion_model_path):
        self.age_model = ort.InferenceSession(age_model_path)
        self.gender_model = ort.InferenceSession(gender_model_path)
        self.emotion_model = ort.InferenceSession(emotion_model_path)
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    async def age(self, file: UploadFile = None, original_image: Image.Image = None):
        input_info = self.age_model.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        output_name = self.age_model.get_outputs()[0].name

        img_h, img_w = input_shape[2], input_shape[3]
        if original_image is None:
            if file is None:
                raise ValueError("Both file and original_image cannot be None.")
            original_image = await read_image(file)

        preprocessed_image = preprocess_image(original_image, (img_w, img_h))

        outputs = self.age_model.run([output_name], {input_name: [preprocessed_image]})
        age = outputs[0][0].argmax() + 1

        return int(age)

    async def gender(self, file: UploadFile = None, original_image: Image.Image = None):
        input_info = self.gender_model.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        output_name = self.gender_model.get_outputs()[0].name

        img_h, img_w = input_shape[2], input_shape[3]
        if original_image is None:
            if file is None:
                raise ValueError("Both file and original_image cannot be None.")
            original_image = await read_image(file)

        preprocessed_image = preprocess_image(original_image, (img_w, img_h))

        outputs = self.gender_model.run([output_name], {input_name: [preprocessed_image]})
        output = outputs[0][0]
        output = 1 / (1 + np.exp(-output))
        gender_label = 'male' if output < 0.5 else 'female'

        return gender_label

    async def emotion(self, file: UploadFile = None, original_image: Image.Image = None):
        input_info = self.emotion_model.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        output_name = self.emotion_model.get_outputs()[0].name

        img_h, img_w = input_shape[2], input_shape[3]
        if original_image is None:
            if file is None:
                raise ValueError("Both file and original_image cannot be None.")
            original_image = await read_image(file, mode='L')
        else:
            original_image = original_image.convert('L')

        preprocessed_image = preprocess_image(original_image, (img_w, img_h))

        outputs = self.emotion_model.run([output_name], {input_name: [preprocessed_image]})
        output = outputs[0][0].argmax()

        emotion_label = self.emotion_labels[output]

        return emotion_label

    async def analyze_face_attributes(self, file: UploadFile):
        original_image = await read_image(file)
        gray_image = original_image.convert('L')

        age_data = await self.age(original_image=original_image)
        gender_data = await self.gender(original_image=original_image)
        emotion_data = await self.emotion(original_image=gray_image)

        attributes_data = {
            'attributes': {
                'age': age_data,
                'gender': gender_data,
                'emotion': emotion_data
            }
        }

        return attributes_data
