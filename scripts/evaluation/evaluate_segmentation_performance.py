import argparse
from ultralytics import YOLO

def evaluate_model(model_path, data_yaml, img_size=640, batch_size=16, plots=False):
    """
    Evaluate model.

    :param model_path: Path to model weights.
    :type model_path: str
    :param data_yaml: Path to the dataset YAML file.
    :type data_yaml: str
    :param img_size: Image size for validation, defaults to 640.
    :type img_size: int, optional
    :param batch_size: Batch size for validation, defaults to 16.
    :type batch_size: int, optional
    :param plots: Whether to generate and save plots, defaults to False.
    :type plots: bool, optional
    :return: Dictionary containing evaluation metrics.
    :rtype: dict
    """
    # Load the model
    model = YOLO(model_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        split='test',
        imgsz=img_size,
        batch=batch_size,
        plots=plots,
        verbose=False
    )
    
    # Extract segmentation metrics
    seg_metrics = results.seg
    
    # Ensure map50 and map metrics are handled properly
    metrics = {
        'mIoU': seg_metrics.maps.mean(),           
        'mIoU_50': seg_metrics.map50.mean(),        
        'precision': seg_metrics.p.mean(),        
        'recall': seg_metrics.r.mean()            
    }
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model performance.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--data-yaml", type=str, required=True, help="Path to the dataset YAML file.")
    parser.add_argument("--img-size", type=int, default=640, help="Image size for validation (default: 640).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for validation (default: 16).")
    parser.add_argument("--plots", action="store_true", help="Generate and save plots.")
    args = parser.parse_args()

    # Run evaluation
    metrics = evaluate_model(
        model_path=args.model_path,
        data_yaml=args.data_yaml,
        img_size=args.img_size,
        batch_size=args.batch_size,
        plots=args.plots
    )

    # Metrics
    print(f"mIoU (0.50-0.95): {metrics['mIoU']:.4f}")
    print(f"mIoU (0.50): {metrics['mIoU_50']:.4f}")
    print(f"Mean Precision: {metrics['precision']:.4f}")
    print(f"Mean Recall: {metrics['recall']:.4f}")
