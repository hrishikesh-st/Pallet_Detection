import os
import yaml
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split
from ultralytics import YOLO


def apply_augmentation(image, labels=None):
    """
    Apply data augmentation to images and bounding boxes.

    :param image: Input image as a NumPy array.
    :type image: np.ndarray
    :param labels: Bounding box labels [class_id, x_center, y_center, width, height], defaults to None.
    :type labels: list[list[float]], optional
    :return: Augmented image and updated labels.
    :rtype: tuple[np.ndarray, list[list[float]]]
    """
    def clip_coordinates(bbox):
        """
        Ensure bounding box coordinates remain within valid bounds.

        :param bbox: Bounding box [x_center, y_center, width, height].
        :type bbox: list[float]
        :return: Clipped bounding box.
        :rtype: list[float]
        """
        x, y, w, h = bbox
        x = max(0, min(1 - w, x))
        y = max(0, min(1 - h, y))
        w = min(w, 1 - x)
        h = min(h, 1 - y)
        return [x, y, w, h]

    # Augmentations
    transform_no_bbox = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=0.2),
    ])

    if labels and len(labels) > 0:
        # Filter and validate bounding boxes
        valid_labels, valid_bboxes, valid_class_labels = [], [], []
        for label in labels:
            try:
                class_id = int(label[0])
                bbox = clip_coordinates(label[1:])
                if bbox[2] > 0 and bbox[3] > 0:  # Validity of dimensions
                    valid_labels.append([class_id] + bbox)
                    valid_bboxes.append(bbox)
                    valid_class_labels.append(class_id)
            except (ValueError, IndexError):
                continue

        if valid_bboxes:
            transform_with_bbox = A.Compose(
                [transform_no_bbox],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
            )
            try:
                transformed = transform_with_bbox(image=image, bboxes=valid_bboxes, class_labels=valid_class_labels)
                new_labels = [[cls, *clip_coordinates(bbox)] for bbox, cls in zip(transformed['bboxes'], transformed['class_labels'])]
                return transformed['image'], new_labels
            except ValueError:
                return transform_no_bbox(image=image)['image'], valid_labels

        return transform_no_bbox(image=image)['image'], valid_labels

    return transform_no_bbox(image=image)['image'], labels


def create_dataset_split(image_dir, label_dir, output_dir, seed=42):
    """
    Split dataset into training, validation, and test sets.

    :param image_dir: Directory containing image files.
    :type image_dir: str
    :param label_dir: Directory containing corresponding label files in YOLO format.
    :type label_dir: str
    :param output_dir: Output directory for split datasets.
    :type output_dir: str
    :param seed: Random seed for reproducibility, defaults to 42.
    :type seed: int, optional
    :return: Path to the dataset YAML file.
    :rtype: str
    """
    np.random.seed(seed)

    image_paths = sorted(list(Path(image_dir).glob("*.jpg")))
    label_dict = {img_path: Path(label_dir) / f"{img_path.stem}.txt" for img_path in image_paths if (Path(label_dir) / f"{img_path.stem}.txt").exists()}

    # Split dataset into train, val, tes sets
    train_val_images, test_images = train_test_split(list(label_dict.keys()), test_size=0.15, random_state=seed)
    train_images, val_images = train_test_split(train_val_images, test_size=0.1765, random_state=seed)

    # Directories for split
    dataset_dir = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def process_split(images, split_name, augment=False):
        for img_path in tqdm(images, desc=f"Processing {split_name} split"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            label_path = label_dict[img_path]
            labels = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    labels = [[float(x) for x in line.strip().split()] for line in f]

            if augment and split_name == 'train' and labels:
                try:
                    img, labels = apply_augmentation(img, labels)
                except Exception as e:
                    print(f"Augmentation failed for {img_path.name}: {e}")

            cv2.imwrite(str(dataset_dir / split_name / 'images' / img_path.name), img)
            with open(dataset_dir / split_name / 'labels' / f"{img_path.stem}.txt", 'w') as f:
                for label in labels:
                    f.write(' '.join(map(str, label)) + '\n')

    # Process split
    process_split(train_images, 'train', augment=True)
    process_split(val_images, 'val', augment=False)
    process_split(test_images, 'test', augment=False)

    # Generate dataset YAML file
    dataset_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {0: 'pallet'},
        'nc': 1
    }
    with open(dataset_dir / 'dataset.yaml', 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    return str(dataset_dir / 'dataset.yaml')


def train_yolo(dataset_yaml, epochs=100, batch_size=16, imgsz=640):
    """
    Train YOLOv8.

    :param dataset_yaml: Path to the dataset YAML file.
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
        model = YOLO('yolov8x.pt')  # Select model variant n, s, m, l, x
        results = model.train(data=dataset_yaml, epochs=epochs, batch=batch_size, imgsz=imgsz, patience=50, save=True, device='0')
        return results
    except ImportError:
        print("Install ultralytics: pip install ultralytics")
        return None


if __name__ == "__main__":
    # Parse cli args
    parser = argparse.ArgumentParser(description="Detection Dataset Preparation and Training")
    parser.add_argument("--image_dir", required=True, help="Path to the directory containing images.")
    parser.add_argument("--label_dir", required=True, help="Path to the directory containing YOLO-format labels.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory for dataset splits.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640).")
    args = parser.parse_args()

    # Dataset prep and train
    dataset_yaml = create_dataset_split(args.image_dir, args.label_dir, args.output_dir, seed=42)
    train_yolo(dataset_yaml, epochs=args.epochs, batch_size=args.batch_size, imgsz=args.imgsz)

