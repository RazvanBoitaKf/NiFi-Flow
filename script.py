import os
import re
import json
from PIL import Image
import cv2
import numpy as np
import shutil

INCOMING_DIR = 'incoming'


def finalize_and_move(source_dir, target_dir):
    for file in os.listdir(source_dir):
        if file.endswith('.jpg') or file.endswith('.txt'):
            shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))


def delete_folder(folder_name):
    target_path = os.path.join(os.getcwd(), folder_name)

    if os.path.exists(target_path) and os.path.isdir(target_path):
        shutil.rmtree(target_path)
        print(f"Deleted folder: {target_path}")
    else:
        print(f"Folder '{folder_name}' does not exist in the current directory.")


def save_yolo_boxes(yolo_boxes, image_path):
    txt_path = os.path.splitext(image_path)[0] + '.txt'

    with open(txt_path, 'w') as f:
        for box in yolo_boxes:
            line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
            f.write(line)

    print(f"Saved YOLO annotations to {txt_path}")

def convert_to_yolo(json_path, image_path):
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    with open(json_path, 'r') as f:
        annotations = json.load(f)

    yolo_annotations = []

    for ann in annotations:
        pos_str = ann.get('positionValue', '')
        if not pos_str.startswith('xywh='):
            continue

        xywh_str = pos_str[len('xywh='):]
        x_str, y_str, w_str, h_str = xywh_str.split(',')

        x, y, w, h = float(x_str), float(y_str), float(w_str), float(h_str)

        center_x = (x + w / 2) / img_w
        center_y = (y + h / 2) / img_h
        norm_w = w / img_w
        norm_h = h / img_h

        yolo_annotations.append([0, center_x, center_y, norm_w, norm_h])

    return yolo_annotations



def group_files(directory):
    files = os.listdir(directory)

    pattern = re.compile(r'^(.*)(annotations\.json|original\.jpg)$', re.IGNORECASE)

    grouped = {}

    for filename in files:
        match = pattern.match(filename)
        if not match:
            continue

        basename, suffix = match.groups()
        basename = basename.strip() 

        if basename not in grouped:
            grouped[basename] = {'json': None, 'image': None}

        full_path = os.path.join(directory, filename)

        if suffix.lower() == 'annotations.json':
            grouped[basename]['json'] = full_path
        elif suffix.lower() == 'original.jpg':
            grouped[basename]['image'] = full_path

    return grouped


def augment_image_variants(image_path, base_name, yolo_boxes, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]

    bright = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
    dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
    high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    low_contrast = cv2.convertScaleAbs(img, alpha=0.6, beta=0)

    shadow_mask = np.ones_like(img, dtype=np.float32)
    shadow_mask[height//3:, width//2:] *= 0.7
    shadowed = (img * shadow_mask).astype(np.uint8)

    augmentations = {
        'bright': bright,
        'dark': dark,
        'high_contrast': high_contrast,
        'low_contrast': low_contrast,
        'shadowed': shadowed
    }

    for aug_name, aug_img in augmentations.items():
        img_filename = f"{base_name}{aug_name}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, aug_img)

        # Create matching .txt file
        txt_filename = f"{base_name}{aug_name}.txt"
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            for box in yolo_boxes:
                line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)

    print(f"Saved augmented images and labels for {base_name}")

if __name__ == "__main__":
    pairs = group_files(INCOMING_DIR)

    for base, files in pairs.items():
        print(f"Base name: {base}")
        print(f"  JSON:  {files['json']}")
        print(f"  Image: {files['image']}")
        yolo_boxes = convert_to_yolo(files['json'], files['image'])
        print(yolo_boxes)
        save_yolo_boxes(yolo_boxes, files['image'])
        augment_image_variants(files['image'], base, yolo_boxes)
    finalize_and_move('incoming/', 'C:/Users/eu/simple_yolo_trainer/datasets/yolo/example/train')
    delete_folder('incoming/')



