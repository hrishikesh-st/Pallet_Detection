import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import argparse
import torch


def process_image(image_path, model_path, save_dir, conf_threshold=0.5):
    """
    Process an image using a YOLO model and save annotated predictions.

    :param image_path: Path to the input image.
    :type image_path: str
    :param model_path: Path to the YOLO model file.
    :type model_path: str
    :param save_dir: Directory to save the annotated image.
    :type save_dir: str
    :param conf_threshold: Confidence threshold for predictions, defaults to 0.5.
    :type conf_threshold: float
    """
    # Load model
    model = YOLO(model_path)

    image = cv2.imread(image_path)

    # Predict
    results = model.predict(
        source=image,
        conf=conf_threshold,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        retina_masks=True,
        verbose=False,
        task='detect'
    )

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f'{cls} {conf:.2f}'
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    os.makedirs(save_dir, exist_ok=True)
    output_image_path = Path(save_dir) / f"pred_{os.path.basename(image_path)}"
    cv2.imwrite(str(output_image_path), image)
    print(f"Annotated image saved at: {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection model on an image and save results.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save annotated predictions (default: 'results').")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold for predictions (default: 0.5).")
    args = parser.parse_args()

    process_image(
        image_path=args.image_path,
        model_path=args.model_path,
        save_dir=args.save_dir,
        conf_threshold=args.conf_threshold
    )