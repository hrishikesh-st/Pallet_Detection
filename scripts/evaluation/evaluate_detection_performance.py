import argparse
from ultralytics import YOLO

def evaluate_model(model_path, data_yaml, img_size=640, batch_size=16, plots=True):
    """
    Evaluate model.

    :param model_path: Path to the YOLO model weights file to be validated.
    :type model_path: str
    :param data_yaml: Path to the dataset YAML file, which contains information about classes and dataset structure.
    :type data_yaml: str
    :param img_size: Image size (in pixels) to use for validation, defaults to 640.
    :type img_size: int, optional
    :param batch_size: Number of images processed per batch during validation, defaults to 16.
    :type batch_size: int, optional
    :param plots: Whether to generate and save validation plots (e.g., confusion matrix, precision-recall curve), defaults to True.
    :type plots: bool, optional
    """
    # Load model
    model = YOLO(model_path)

    results = model.val(
        data=data_yaml,
        imgsz=img_size,
        split='test',
        batch=batch_size,
        plots=plots, 
    )

    # AMetrics
    print(f"mAP50: {results.box.map50}")     
    print(f"mAP50-95: {results.box.map}")    
    print(f"Precision: {results.box.p}")     
    print(f"Recall: {results.box.r}")         


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate detection model performance.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLO model weights file.")
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for validation (default: 640).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for validation (default: 16).")
    parser.add_argument("--plots", action="store_true", help="Generate and save plots.")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        img_size=args.img_size,
        batch_size=args.batch_size,
        plots=args.plots
    )
