import argparse
from ultralytics import YOLO

def export_model(model_path, export_format="engine", half_precision=True):
    model = YOLO(model_path)
    model.export(format=export_format, half=half_precision)
    print(f"Model exported successfully to {export_format} format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export pytorch to tensorrt format.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the YOLO model weights file.")
    parser.add_argument("--export-format", type=str, default="engine", help="Format to export the model (default: 'engine').")
    parser.add_argument("--half", action="store_true", help="Export the model in half precision (default: True).")
    args = parser.parse_args()

    export_model(
        model_path=args.model_path,
        export_format=args.export_format,
        half_precision=args.half
    )
