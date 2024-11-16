import os
import yaml
import argparse
from ultralytics import YOLO


def create_dataset_yaml(dataset_path, output_path):
    """
    Create a YAML configuration file for a segmentation dataset.

    :param dataset_path: Path to the root dataset directory.
    :type dataset_path: str
    :param output_path: Path to save the YAML configuration file.
    :type output_path: str
    :return: Path to the saved YAML configuration file.
    :rtype: str
    """
    dataset_yaml = {
        'path': dataset_path,  # Root dir
        'train': 'train/images',  
        'val': 'val/images',      
        'test': 'test/images',    
        'train_masks': 'train/labels',  
        'val_masks': 'val/labels',      
        'test_masks': 'test/labels',    
        'names': {0: 'floor', 1: 'pallet'},
        'nc': 2,  
        'task': 'segment'  # Task specification
    }
    
    # Save YAML file
    with open(output_path, 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)
    
    return output_path


def train_yolo_seg(dataset_yaml, epochs=100, batch_size=16, imgsz=640):
    """
    Train a YOLOv8 segmentation model.

    :param dataset_yaml: Path to the dataset YAML configuration file.
    :type dataset_yaml: str
    :param epochs: Number of training epochs, defaults to 100.
    :type epochs: int, optional
    :param batch_size: Training batch size, defaults to 16.
    :type batch_size: int, optional
    :param imgsz: Image size for training, defaults to 640.
    :type imgsz: int, optional
    :return: Training results object.
    :rtype: ultralytics.engine.results.Results
    """
    try:
        print("\nInitializing YOLOv8-seg model...")
        model = YOLO('yolov8x-seg.pt')  # Load model variant (s, m, l, x)
        
        print("\nStarting segmentation training...")
        results = model.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            patience=50,
            save=True,
            device='0',
            task='segment'  # Task specification
        )
        
        print("\nTraining completed successfully!")
        return results
        
    except ImportError:
        print("Install ultralytics: pip install ultralytics")
        return None


if __name__ == "__main__":
    # Parse cli args
    parser = argparse.ArgumentParser(description="Segmentation Training")
    parser.add_argument("--dataset_path", required=True, help="Path to the root dataset directory.")
    parser.add_argument("--yaml_path", required=True, help="Path to save the dataset YAML configuration file.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640).")
    args = parser.parse_args()

    # YAML file
    dataset_yaml = create_dataset_yaml(args.dataset_path, args.yaml_path)
    
    # Train
    train_yolo_seg(dataset_yaml, epochs=args.epochs, batch_size=args.batch_size, imgsz=args.imgsz)
