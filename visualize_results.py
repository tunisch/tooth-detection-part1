import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results_file, save_path="results_combined.png"):
    """
    Plots training and validation metrics from the results CSV file in a single figure with subplots.
    Args:
        results_file (str): Path to the results CSV file generated during training.
        save_path (str): Path to save the combined results plot as a PNG file.
    """
    # Load results from CSV
    df = pd.read_csv(results_file, delimiter=',')  # CSV dosyasını oku

    # Clean column names by removing extra spaces
    df.columns = df.columns.str.strip()

    # Extract metrics
    epoch = df['epoch']
    train_box_loss = df['train/box_loss']
    train_obj_loss = df['train/obj_loss']
    train_cls_loss = df['train/cls_loss']
    val_box_loss = df['val/box_loss']
    val_obj_loss = df['val/obj_loss']
    val_cls_loss = df['val/cls_loss']
    precision = df['metrics/precision']
    recall = df['metrics/recall']
    mAP_0_5 = df['metrics/mAP_0.5']
    mAP_0_5_0_95 = df['metrics/mAP_0.5:0.95']

    # Create subplots
    fig, axs = plt.subplots(2, 5, figsize=(20, 10))
    axs = axs.ravel()  # Flatten the 2D array of axes for easier indexing

    # Plot each metric
    metrics = [
        (train_box_loss, "train/box_loss"),
        (train_obj_loss, "train/obj_loss"),
        (train_cls_loss, "train/cls_loss"),
        (precision, "metrics/precision"),
        (recall, "metrics/recall"),
        (val_box_loss, "val/box_loss"),
        (val_obj_loss, "val/obj_loss"),
        (val_cls_loss, "val/cls_loss"),
        (mAP_0_5, "metrics/mAP_0.5"),
        (mAP_0_5_0_95, "metrics/mAP_0.5:0.95"),
    ]

    for i, (metric, title) in enumerate(metrics):
        axs[i].plot(epoch, metric, label=title, marker='o')
        axs[i].set_title(title)
        axs[i].set_xlabel("Epochs")
        axs[i].set_ylabel("Value")
        axs[i].legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(save_path)  # Save as PNG
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    results_file = "runs/train/exp_640px_500ep_s/results.csv"  # Update this path if needed
    plot_results(results_file)