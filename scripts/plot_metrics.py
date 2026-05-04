import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot training curves from logs.json.txt")
    parser.add_argument("log_file", type=str, help="Path to the logs.json.txt file (inside data/outputs/...)")
    parser.add_argument("--save", type=str, default=None, help="Optional: save the plot to an image instead of displaying it")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"File not found: {log_path}")
        return

    epochs = []
    train_loss = []
    val_loss = []
    val_epochs = []

    print(f"Reading {log_path}...")
    with open(log_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                # Training loss
                if 'train_loss' in data:
                    epochs.append(data.get('epoch', 0))
                    train_loss.append(data['train_loss'])
                
                # Validation loss (if presents)
                if 'val_loss' in data:
                    val_epochs.append(data.get('epoch', 0))
                    val_loss.append(data['val_loss'])
            except:
                pass


    plt.figure(figsize=(10, 6))
    if train_loss:
        plt.plot(epochs, train_loss, label='Train Loss', alpha=0.7)
    if val_loss:
        # Plot val loss usually as scatter or line depending on frequency
        plt.plot(val_epochs, val_loss, label='Val Loss', color='orange', marker='o')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Curves')
    plt.legend()
    plt.grid(True)

    if args.save:
        plt.savefig(args.save)
        print(f"Saved plot to {args.save}")
    else:
        # On a headless server this might crash, we suggest using --save
        try:
            plt.show()
        except:
            print("Could not display plot. If you're on a server, try running with --save plot.png")
            fallback = log_path.parent / "curves.png"
            plt.savefig(fallback)
            print(f"Saved fallback plot to {fallback}")

if __name__ == "__main__":
    main()
