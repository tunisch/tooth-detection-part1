import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results_file):
    """
    Plots training and validation metrics from the results CSV file.
    Args:
        results_file (str): Path to the results CSV file generated during training.
    """
    if not os.path.exists(results_file):
        print(f"Error: {results_file} does not exist.")
        return

    # Load results from CSV
    df = pd.read_csv(results_file, delimiter=',')
    
    # Clean column names by removing extra spaces
    df.columns = df.columns.str.strip()

    # Extract metrics
    epoch = df['epoch']
    train_loss = df['train/box_loss']
    val_loss = df['val/box_loss']
    precision = df['metrics/precision']
    recall = df['metrics/recall']
    mAP_50 = df['metrics/mAP_0.5']
    mAP_50_95 = df['metrics/mAP_0.5:0.95']

    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(epoch, train_loss, label='Train Box Loss', color='blue')
    plt.plot(epoch, val_loss, label='Validation Box Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot precision, recall, and mAP
    plt.figure(figsize=(12, 6)) 
    plt.plot(epoch, precision, label='Precision', color='green')
    plt.plot(epoch, recall, label='Recall', color='red')
    plt.plot(epoch, mAP_50, label='mAP@0.5', color='purple')
    plt.plot(epoch, mAP_50_95, label='mAP@0.5:0.95', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Precision, Recall, and mAP')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Update the path to your results.csv file
    results_file = "C:/Users/tunah/Desktop/yolo-deneme-2/yolov5/runs/train/exp_640px_hyp_tuned_s/results.csv"
    plot_results(results_file)