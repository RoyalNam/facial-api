import torch
import torch.nn as nn
from fastapi.responses import StreamingResponse
from fastapi import UploadFile
from utils import *
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
from controllers.attributesController import FaceAttributes


class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)


faceAttributes = FaceAttributes(
    'model/face-age.onnx',
    'model/face-gender.onnx',
    'model/face-emotion.onnx',
)


async def genAttributes(file: UploadFile):
    img_size = (128, 128)
    model = Generator()
    model.load_state_dict(torch.load('model/face-gen-attrs.pth', map_location=torch.device('cpu')))

    origin_image = await read_image(file)
    gender = await faceAttributes.gender(original_image=origin_image)

    preprocessed_image = preprocess_image(origin_image, img_size)
    preprocessed_image = denormalize(preprocessed_image)
    preprocessed_image = torch.tensor(preprocessed_image).unsqueeze(0)

    # Number of attributes
    attributes = ['Black Hair', 'Blond Hair', 'Brown Hair', 'Male', 'Young']
    num_attributes = len(attributes)

    images_with_text = []

    for i in range(num_attributes):
        c = torch.zeros(1, num_attributes)
        c[0, i] = 1

        if gender == 'male':
            if attributes[i] == 'Male':
                c[0, -2] = 0
            else:
                c[0, -2] = 1

        with torch.no_grad():
            output_image = model(preprocessed_image, c).squeeze(0)

        output_image = normalize(output_image)
        output_image = (output_image * 255).byte().numpy()
        output_image = output_image.transpose(1, 2, 0)
        output_image = Image.fromarray(output_image)

        # Create a new blank image with black background
        text_image = Image.new('RGB', (output_image.width, output_image.height + 20), color=(0, 0, 0))

        # Paste the generated image on the black background image
        text_image.paste(output_image, (0, 20))

        # Draw text on the black background image
        draw = ImageDraw.Draw(text_image)
        draw.text((5, 5), attributes[i], fill=(255, 255, 255))

        images_with_text.append(text_image)

    # Concatenate images horizontally using numpy
    concatenated_image = np.concatenate([np.array(img) for img in images_with_text], axis=1)
    concatenated_image = Image.fromarray(concatenated_image)

    buf = BytesIO()
    concatenated_image.save(buf, format='PNG')
    buf.seek(0)

    return StreamingResponse(buf, media_type='image/png')
