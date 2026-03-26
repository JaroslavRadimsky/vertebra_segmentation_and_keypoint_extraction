import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_loss_json(path: str):
    """
    JSON format:
    [
      [timestamp, epoch, loss],
      ...
    ]
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    # seřadit podle epochy
    data = sorted(data, key=lambda x: x[1])

    epochs = [int(row[1]) for row in data]
    loss = [float(row[2]) for row in data]
    return epochs, loss


def plot_loss(
    epochs,
    loss,
    title,
    output_path,
    y_min=0.0,
    y_max=1.0,
):
    plt.figure()
    plt.plot(epochs, loss, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(y_min, y_max)   # 👈 rozsah Y osy
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    # === CESTY K SOUBORŮM ===
    train_path = "loss_train.json"
    val_path = "loss_val.json"

    # === NASTAVENÍ ROZSAHU Y OSY ===
    y_min = 0.05
    y_max = 0.1

    # === NAČTENÍ DAT ===
    train_epochs, train_loss = load_loss_json(train_path)
    val_epochs, val_loss = load_loss_json(val_path)

    # === TRAIN GRAF ===
    plot_loss(
        train_epochs,
        train_loss,
        title="Train Loss během tréninku",
        output_path="loss_train.png",
        y_min=y_min,
        y_max=y_max,
    )

    # === VALIDATION GRAF ===
    plot_loss(
        val_epochs,
        val_loss,
        title="Validation Loss během tréninku",
        output_path="loss_val.png",
        y_min=y_min,
        y_max=y_max,
    )

    print("Hotovo ✅")
    print("Uloženo:")
    print(" - loss_train.png")
    print(" - loss_val.png")


if __name__ == "__main__":
    main()
