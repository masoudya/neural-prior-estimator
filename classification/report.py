import numpy as np
import argparse
import matplotlib.pyplot as plt


IGNORE_KEYS = {
    "gpu", "dist_backend", "dist_url", "distributed",
    "multiprocessing_distributed", "rank", "resume",
    "save_checkpoint", "workers", "world_size", "print_freq"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to run_*.npy")
    parser.add_argument("--visualize-train", action="store_true",
                        help="Plot training loss curve")
    parser.add_argument("--visualize-test", action="store_true",
                        help="Plot validation accuracy and group accuracies")
    return parser.parse_args()


def compute_group_accs(exp_log):
    cfg = exp_log["config"]
    final_per_class = exp_log["per_class_accs"][-1]   # shape [num_classes]

    def avg(idx_pair):
        start, end = idx_pair
        return float(final_per_class[start:end].mean())

    head_acc = avg(cfg["head_class_idx"])
    med_acc = avg(cfg["med_class_idx"])
    tail_acc = avg(cfg["tail_class_idx"])

    return head_acc, med_acc, tail_acc


def summarize(file_path):
    try:
        exp_log = np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"❌ Could not read '{file_path}': {e}")
    
    cfg = exp_log["config"]
    final_acc = float(exp_log["val_accs"][-1])

    head_acc, med_acc, tail_acc = compute_group_accs(exp_log)

    report_keys = [
        "backbone", "dataset", "imb_type", "imb_factor", 
        "loss_function","num_ach",
        "num_epochs", "lr", "momentum", "weight_decay",
        
    ]

    print("=" * 60)
    print(" EXPERIMENT REPORT")
    print("=" * 60)
    print(f"Timestamp         : {exp_log['timestamp']}")
    print("-" * 60)

    for k in report_keys:
        if k in cfg:
            print(f"{k:<15}: {cfg[k]}")

    print("-" * 60)
    print(f"Final Accuracy    : {final_acc:.4f}")
    print(f"Head Accuracy     : {head_acc:.4f}")
    print(f"Medium Accuracy   : {med_acc:.4f}")
    print(f"Tail Accuracy     : {tail_acc:.4f}")
    print("=" * 60)
    
    return exp_log

def plot_train(exp_log):
    plt.figure(figsize=(8, 5))
    plt.plot(exp_log["train_losses"], label="Train Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_test(exp_log):
    # Validation accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(exp_log["val_accs"], label="Validation Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Group accuracies (head/med/tail)
    head_acc, med_acc, tail_acc = compute_group_accs(exp_log)
    plt.figure(figsize=(6, 4))
    groups = ["Head", "Medium", "Tail"]
    values = [head_acc, med_acc, tail_acc]

    plt.bar(groups, values)
    plt.title("Final Group Accuracies")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    exp_log = summarize(args.path)

    if args.visualize_train:
        plot_train(exp_log)

    if args.visualize_test:
        plot_test(exp_log)


if __name__ == "__main__":
    main()
