from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO


async def read_image(file, mode='RGB'):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    if mode == 'L':
        if image.mode != 'L':
            image = image.convert('L')
    else:
        if image.mode != 'RGB':
            image = image.convert('RGB')

    return image


def preprocess_image(image, image_size):
    image = image.resize(image_size)
    image = np.array(image)

    if len(image.shape) == 3:
        image = np.transpose(image, [2, 0, 1])
    elif len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)
    else:
        raise ValueError("Unsupported image format")
    image = np.array(image).astype('float32') / 255.

    return image


def normalize(output_image):
    # [-1, 1] -> [0, 1]
    normalized_image = (output_image + 1) / 2
    return normalized_image


def denormalize(normalized_image):
    # [0, 1] -> [-1, 1]
    denormalized_image = normalized_image * 2 - 1
    return denormalized_image


def draw_box(image, targets, color=(0, 255, 0), image_size=(640, 640)):
    draw = ImageDraw.Draw(image)
    w, h = image_size
    img_w, img_h = image.size
    for target in targets:
        x_center, y_center, box_w, box_h = target
        x1 = int((x_center - box_w / 2) * img_w / w)
        y1 = int((y_center - box_h / 2) * img_h / h)
        x2 = int((x_center + box_w / 2) * img_w / w)
        y2 = int((y_center + box_h / 2) * img_h / h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return image


def get_boxes(image, targets, image_size=(640, 640)):
    w, h = image_size
    img_w, img_h = image.size
    boxes = {'xyxy': [], 'xywh': [], 'xywhc': []}
    for target in targets:
        x_center, y_center, box_w, box_h = target
        # xyxy
        x1 = (x_center - box_w / 2) * img_w / w
        y1 = (y_center - box_h / 2) * img_h / h
        x2 = (x_center + box_w / 2) * img_w / w
        y2 = (y_center + box_h / 2) * img_h / h
        boxes['xyxy'].append((x1, y1, x2, y2))

        box_w = x2 - x1
        box_h = y2 - y1
        # xywh
        boxes['xywh'].append((x1, y1, box_w, box_h))

        # xywhc
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        boxes['xywhc'].append((cx, cy, box_w, box_h))

    return boxes


def get_landmarks(image, points, labels, image_size=(320, 320)):
    w, h = image_size
    img_w, img_h = image.size
    adjusted_points = {}
    for i in range(0, len(points), 2):
        x = points[i] * img_w / w
        y = points[i + 1] * img_h / h
        label = labels[i // 2]
        adjusted_points[label] = [x, y]
    return adjusted_points


def draw_landmarks(image, points, color=(0, 0, 255), image_size=(320, 320)):
    draw = ImageDraw.Draw(image)
    w, h = image_size
    img_w, img_h = image.size

    for i in range(0, len(points), 2):
        x = points[i] * img_w / w
        y = points[i+1] * img_h / h
        draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    return image


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1_tl, y1_tl, w1, h1 = box1
    x2_tl, y2_tl, w2, h2 = box2

    # Calculate bottom-right coordinates of the boxes
    x1_br, y1_br = x1_tl + w1, y1_tl + h1
    x2_br, y2_br = x2_tl + w2, y2_tl + h2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1_tl, x2_tl)
    y_intersection = max(y1_tl, y2_tl)
    w_intersection = max(0, min(x1_br, x2_br) - x_intersection)
    h_intersection = max(0, min(y1_br, y2_br) - y_intersection)

    # Calculate area of intersection rectangle
    intersection_area = w_intersection * h_intersection

    # Calculate areas of both bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    union_area = area1 + area2 - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def NMSBoxes(boxes, confidences, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Perform non-maximum suppression (NMS) on a list of bounding boxes.
    """
    # Initialize list to keep track of selected boxes
    selected_boxes = []

    # Sort boxes by confidence score in descending order
    sorted_indices = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)

    # Loop through all boxes
    for i in range(len(boxes)):
        # If confidence score of the current box is below threshold, skip it
        if confidences[i] < confidence_threshold:
            continue

        # Add current box to the list of selected boxes
        selected_boxes.append(boxes[i])

        # Loop through remaining boxes and suppress boxes with high IoU
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > nms_threshold:
                # If IoU is higher than threshold, suppress the box
                confidences[j] = 0

    return selected_boxes
