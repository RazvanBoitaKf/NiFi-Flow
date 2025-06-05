import os
import re
import json
from PIL import Image
import cv2
import numpy as np
import shutil
import logging
from datetime import datetime
import traceback

INCOMING_DIR = 'incoming'
LOG_FILE = 'processing_errors.txt'

# Set up logging
def setup_logging():
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler()  # Also print to console
        ]
    )

def log_error(function_name, error, additional_info=None):
    """Log error details to file and console"""
    error_msg = f"Error in {function_name}: {str(error)}"
    if additional_info:
        error_msg += f" | Additional info: {additional_info}"
    
    logging.error(error_msg)
    logging.error(f"Traceback: {traceback.format_exc()}")
    
    # Also write a separator for readability
    with open(LOG_FILE, 'a') as f:
        f.write("-" * 80 + "\n")

def finalize_and_move(source_dir, target_dir, processed_bases):
    """Only move files that were successfully processed"""
    try:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        for file in os.listdir(source_dir):
            if file.endswith('.jpg') or file.endswith('.txt'):
                try:
                    # Check if this file belongs to a successfully processed base
                    file_base = None
                    
                    # Check if it's an original file or augmented file
                    if file.endswith('original.jpg'):
                        file_base = file.replace('original.jpg', '').strip()
                    elif file.endswith('.txt'):
                        # For txt files, remove the .txt extension and any augmentation suffix
                        name_without_ext = file.replace('.txt', '')
                        # Check if it's an augmented file
                        augmentation_suffixes = ['bright', 'dark', 'high_contrast', 'low_contrast', 'shadowed']
                        for suffix in augmentation_suffixes:
                            if name_without_ext.endswith(suffix):
                                file_base = name_without_ext.replace(suffix, '').strip()
                                break
                        if file_base is None:
                            # It's probably the original txt file
                            file_base = name_without_ext.strip()
                    elif file.endswith('.jpg'):
                        # For jpg files, check if it's an augmented file
                        name_without_ext = file.replace('.jpg', '')
                        augmentation_suffixes = ['bright', 'dark', 'high_contrast', 'low_contrast', 'shadowed']
                        for suffix in augmentation_suffixes:
                            if name_without_ext.endswith(suffix):
                                file_base = name_without_ext.replace(suffix, '').strip()
                                break
                        if file_base is None and file.endswith('original.jpg'):
                            file_base = name_without_ext.replace('original', '').strip()
                    
                    # Only move if the base was successfully processed
                    if file_base and file_base in processed_bases:
                        source_path = os.path.join(source_dir, file)
                        target_path = os.path.join(target_dir, file)
                        shutil.move(source_path, target_path)
                        print(f"Moved: {file}")
                    else:
                        print(f"Skipped moving {file} (base '{file_base}' not in processed list)")
                        
                except Exception as e:
                    log_error("finalize_and_move", e, f"Failed to move file: {file}")
                    
    except Exception as e:
        log_error("finalize_and_move", e, f"Failed to process directory: {source_dir}")

def delete_folder(folder_name):
    try:
        target_path = os.path.join(os.getcwd(), folder_name)

        if os.path.exists(target_path) and os.path.isdir(target_path):
            shutil.rmtree(target_path)
            print(f"Deleted folder: {target_path}")
        else:
            print(f"Folder '{folder_name}' does not exist in the current directory.")
    except Exception as e:
        log_error("delete_folder", e, f"Failed to delete folder: {folder_name}")

def save_yolo_boxes(yolo_boxes, image_path):
    try:
        txt_path = os.path.splitext(image_path)[0] + '.txt'

        with open(txt_path, 'w') as f:
            for box in yolo_boxes:
                line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                f.write(line)

        print(f"Saved YOLO annotations to {txt_path}")
        return True
    except Exception as e:
        log_error("save_yolo_boxes", e, f"Failed to save annotations for: {image_path}")
        return False

def convert_to_yolo(json_path, image_path):
    try:
        # Try to open and get image dimensions
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            log_error("convert_to_yolo", e, f"Failed to open image: {image_path}")
            return []

        # Try to read JSON file
        try:
            with open(json_path, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            log_error("convert_to_yolo", e, f"Failed to read JSON: {json_path}")
            return []

        yolo_annotations = []

        for i, ann in enumerate(annotations):
            try:
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
                
            except Exception as e:
                log_error("convert_to_yolo", e, f"Failed to process annotation {i} in {json_path}")
                continue  # Skip this annotation but continue with others

        return yolo_annotations
        
    except Exception as e:
        log_error("convert_to_yolo", e, f"General error processing {json_path} and {image_path}")
        return []

def group_files(directory):
    try:
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return {}
            
        files = os.listdir(directory)
        pattern = re.compile(r'^(.*)(annotations\.json|original\.jpg)$', re.IGNORECASE)
        grouped = {}

        for filename in files:
            try:
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
                    
            except Exception as e:
                log_error("group_files", e, f"Failed to process filename: {filename}")
                continue

        return grouped
        
    except Exception as e:
        log_error("group_files", e, f"Failed to process directory: {directory}")
        return {}

def augment_image_variants(image_path, base_name, yolo_boxes, output_dir=None):
    try:
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        
        img = cv2.imread(image_path)
        if img is None:
            log_error("augment_image_variants", "Image is None", f"Could not read image: {image_path}")
            return False

        height, width = img.shape[:2]

        try:
            # Create augmentations
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

            successful_augmentations = 0
            
            for aug_name, aug_img in augmentations.items():
                try:
                    img_filename = f"{base_name}{aug_name}.jpg"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    # Save augmented image
                    if cv2.imwrite(img_path, aug_img):
                        # Create matching .txt file
                        txt_filename = f"{base_name}{aug_name}.txt"
                        txt_path = os.path.join(output_dir, txt_filename)

                        with open(txt_path, 'w') as f:
                            for box in yolo_boxes:
                                line = f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n"
                                f.write(line)
                        
                        successful_augmentations += 1
                    else:
                        log_error("augment_image_variants", "cv2.imwrite failed", f"Could not save {img_path}")
                        
                except Exception as e:
                    log_error("augment_image_variants", e, f"Failed to create {aug_name} variant for {base_name}")
                    continue

            print(f"Saved {successful_augmentations}/5 augmented images and labels for {base_name}")
            return successful_augmentations > 0
            
        except Exception as e:
            log_error("augment_image_variants", e, f"Failed to create augmentations for {image_path}")
            return False
            
    except Exception as e:
        log_error("augment_image_variants", e, f"General error in augmentation for {image_path}")
        return False

if __name__ == "__main__":
    # Initialize logging
    setup_logging()
    
    # Log start of processing
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Processing started at: {datetime.now()}\n")
        f.write(f"{'='*80}\n")
    
    print(f"Starting processing... Errors will be logged to: {LOG_FILE}")
    
    try:
        pairs = group_files(INCOMING_DIR)
        
        if not pairs:
            print("No valid file pairs found to process")
        else:
            processed_count = 0
            error_count = 0
            successfully_processed_bases = set()  # Track which bases were fully processed
            
            for base, files in pairs.items():
                try:
                    if files['json'] is None:
                        print(f"Skipping {base}: No JSON file found")
                        continue
                        
                    if files['image'] is None:
                        print(f"Skipping {base}: No image file found")
                        continue
                    
                    print(f"Processing base name: {base}")
                    print(f"  JSON:  {files['json']}")
                    print(f"  Image: {files['image']}")
                    
                    yolo_boxes = convert_to_yolo(files['json'], files['image'])
                    
                    if not yolo_boxes:
                        print(f"No valid YOLO boxes found for {base}")
                        error_count += 1
                        continue
                    
                    print(f"Found {len(yolo_boxes)} YOLO boxes")
                    
                    # Try to save YOLO boxes
                    if save_yolo_boxes(yolo_boxes, files['image']):
                        # Try to create augmentations
                        if augment_image_variants(files['image'], base, yolo_boxes):
                            processed_count += 1
                            successfully_processed_bases.add(base)  # Only add if everything succeeded
                            print(f"Successfully processed {base}")
                        else:
                            error_count += 1
                            print(f"Failed to create augmentations for {base}")
                    else:
                        error_count += 1
                        print(f"Failed to save YOLO boxes for {base}")
                        
                except Exception as e:
                    error_count += 1
                    log_error("main_processing_loop", e, f"Failed to process base: {base}")
                    print(f"Error processing {base}, continuing with next...")
                    continue
            
            print(f"\nProcessing summary:")
            print(f"Successfully processed: {processed_count}")
            print(f"Errors encountered: {error_count}")
            print(f"Successfully processed bases: {list(successfully_processed_bases)}")
            
        # Try to finalize and move files - only move files from successfully processed bases
        print("\nFinalizing and moving files...")
        finalize_and_move('incoming/', 'C:/Users/eu/simple_yolo_trainer/datasets/yolo/example/train', successfully_processed_bases)
        
        # Try to delete folder
        print("Cleaning up...")
        delete_folder('incoming/')
        
    except Exception as e:
        log_error("main", e, "Critical error in main execution")
        print("Critical error occurred. Check the log file for details.")
    
    print(f"Processing completed. Check {LOG_FILE} for any errors that occurred.")