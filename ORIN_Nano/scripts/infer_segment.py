import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


color = {
    0: (46, 139, 87),    
    1: (255, 69, 0),     
    'overlay_alpha': 0.4,
    'contour_thickness': 2,
    'floor_contour': (0, 255, 0),
    'pallet_contour': (0, 69, 255)
}


def create_visualization(image, masks, classes, confidences):
    """Create visualization with different colors per class."""
    if image is None or not masks:
        return None

    overlay = image.copy()

    # Process each instance
    for mask, class_id, confidence in zip(masks, classes, confidences):
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color[int(class_id)]

        overlay = cv2.addWeighted(
            overlay,
            1,
            colored_mask,
            color['overlay_alpha'],
            0
        )

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contour_color = color['floor_contour'] if int(class_id) == 0 else color['pallet_contour']

        cv2.drawContours(
            overlay,
            contours,
            -1,
            contour_color,
            color['contour_thickness']
        )

        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            class_name = "Floor" if int(class_id) == 0 else "Pallet"
            text = f'{class_name} {confidence:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            # Add text
            cv2.putText(
                overlay,
                text,
                (cX - 60, cY),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )

    return overlay


def process_segmentation(image_path, model_path, save_dir, conf_threshold=0.5):
    """
    Process an image using a YOLO segmentation model and save annotated results.
    """
    # Load model
    model = YOLO(model_path, task="segment")

    image = cv2.imread(image_path)

    # Predict
    results = model.predict(
        source=image,
        conf=conf_threshold,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        retina_masks=True,
        verbose=False,
        task='segment'
    )

    for result in results:
        if result.masks is not None:
            masks = result.masks.data
            boxes = result.boxes.data
            classes = result.boxes.cls.cpu().numpy()
            confidences = boxes[:, 4].cpu().numpy()

            masks_np = [mask.cpu().numpy().astype(bool) for mask in masks]

            visualization = create_visualization(
                image,
                masks_np,
                classes,
                confidences
            )

            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)

                    if visualization is not None:
                        output_path = Path(save_dir) / f"pred_{Path(image_path).name}"
                        cv2.imwrite(str(output_path), visualization)

                        combined_view = np.hstack((image, visualization))
                        comparison_path = Path(save_dir) / f"comparison_{Path(image_path).name}"
                        cv2.imwrite(str(comparison_path), combined_view)

                        print(f"Saved results to {save_dir}")

                except Exception as e:
                    print(f"Error saving results for {image_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation model on an image and save results.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save annotated results.")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for predictions (default: 0.5).")
    args = parser.parse_args()

    process_segmentation(
        image_path=args.image_path,
        model_path=args.model_path,
        save_dir=args.save_dir,
        conf_threshold=args.conf_threshold
    )