#!/usr/bin/env python3

import cv2
import numpy as np

def draw_detections(image, predictions, ratio, class_names, conf_threshold=0.45):
    """
    Draw detection boxes and labels on the image.
    
    Args:
        image: Original image (BGR format)
        predictions: YOLOX predictions after NMS
        ratio: Scale ratio from preprocessing
        class_names: Tuple of class names
        conf_threshold: Confidence threshold for displaying detections
    
    Returns:
        Image with drawn detections
    """
    result_image = image.copy()

    if predictions is None:
        return result_image

    # Convert predictions to numpy and rescale boxes
    predictions = predictions.cpu().numpy()
    boxes = predictions[:, :4] / ratio
    scores = predictions[:, 4] * predictions[:, 5]  # obj_conf * cls_conf
    class_ids = predictions[:, 6].astype(int)

    # Draw each detection
    for box, score, class_id in zip(boxes, scores, class_ids):
        if score < conf_threshold:
            continue

        # Get box coordinates
        x1, y1, x2, y2 = box.astype(int)

        # Draw bounding box
        cv2.rectangle(result_image, 
                    (x1, y1), 
                    (x2, y2), 
                    (0, 255, 0),  # Green color
                    2)  # Thickness

        # Create label with class name and score
        label = f'{class_names[class_id]}: {score:.2f}'
        
        # Get label size for background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale
            1    # Thickness
        )

        # Draw filled background rectangle for label
        cv2.rectangle(result_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1)  # Filled rectangle

        # Draw label text
        cv2.putText(result_image,
                  label,
                  (x1, y1 - baseline - 5),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.5,  # Font scale
                  (0, 0, 0),  # Black text
                  1)   # Thickness

    return result_image

def draw_fps(image, fps):
    """
    Draw FPS counter on the image.
    
    Args:
        image: Input image
        fps: FPS value to display
    
    Returns:
        Image with FPS counter
    """
    # Draw FPS in top-left corner
    cv2.putText(image,
                f"FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)
    return image