import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse


def plot_training_metrics(csv_path, output_dir='training_plots'):
    """
    Read training metrics from CSV and create plots.
    Args:
        csv_path: Path to the results CSV file.
        output_dir: Directory to save the plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    results_df = pd.read_csv(csv_path)
    
    # Extract metrics
    epochs = results_df['epoch']
    
    # Calculate total loss (box_loss + cls_loss + dfl_loss)
    train_total_loss = (results_df['train/box_loss'] + 
                        results_df['train/cls_loss'] + 
                        results_df['train/dfl_loss'])
    val_total_loss = (results_df['val/box_loss'] + 
                      results_df['val/cls_loss'] + 
                      results_df['val/dfl_loss'])
    
    # Create figure and subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Total Loss 
    ax1.plot(epochs, train_total_loss, 'b-', label='Training Loss')
    ax1.plot(epochs, val_total_loss, 'r-', label='Validation Loss')
    ax1.set_title('Total Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Component-wise Loss 
    ax2.plot(epochs, results_df['train/box_loss'], 'b-', label='Box Loss')
    ax2.plot(epochs, results_df['train/cls_loss'], 'g-', label='Class Loss')
    ax2.plot(epochs, results_df['train/dfl_loss'], 'r-', label='DFL Loss')
    ax2.set_title('Training Loss Components')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # mAP 
    ax3.plot(epochs, results_df['metrics/mAP50(B)'], 'b-', label='mAP50')
    ax3.plot(epochs, results_df['metrics/mAP50-95(B)'], 'r-', label='mAP50-95')
    ax3.set_title('Mean Average Precision (mAP)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('mAP')
    ax3.legend()
    ax3.grid(True)
    
    # Precision-Recall
    ax4.plot(epochs, results_df['metrics/precision(B)'], 'b-', label='Precision')
    ax4.plot(epochs, results_df['metrics/recall(B)'], 'r-', label='Recall')
    ax4.set_title('Precision and Recall')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning rate plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, results_df['lr/pg0'], 'b-', label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from a YOLO training run.")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to the results CSV file.")
    parser.add_argument("--output-dir", type=str, default='training_plots', help="Directory to save the plots.")
    args = parser.parse_args()
    
    plot_training_metrics(csv_path=args.csv_path, output_dir=args.output_dir)
