import json
from pathlib import Path
import matplotlib.pyplot as plt


def load_dice_json(path: str):
    """
    JSON format:
    [
      [timestamp, epoch, dice],
      ...
    ]
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    # seřadit podle epochy
    data = sorted(data, key=lambda x: x[1])

    epochs = [int(row[1]) for row in data]
    dice = [float(row[2]) for row in data]
    return epochs, dice


def plot_dice(
    epochs,
    dice,
    title,
    output_path,
    y_min=0.0,
    y_max=1.0,
):
    plt.figure()
    plt.plot(epochs, dice, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("DICE")
    plt.ylim(y_min, y_max)   # 👈 rozsah Y osy
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main():
    # === CESTY K SOUBORŮM ===
    train_path = "dice_train.json"
    val_path = "dice_val.json"

    # === NASTAVENÍ ROZSAHU Y OSY ===
    y_min = 0.9
    y_max = 0.96

    # === NAČTENÍ DAT ===
    train_epochs, train_dice = load_dice_json(train_path)
    val_epochs, val_dice = load_dice_json(val_path)

    # === TRAIN GRAF ===
    plot_dice(
        train_epochs,
        train_dice,
        title="Train DICE během tréninku",
        output_path="dice_train.png",
        y_min=y_min,
        y_max=y_max,
    )

    # === VALIDATION GRAF ===
    plot_dice(
        val_epochs,
        val_dice,
        title="Validation DICE během tréninku",
        output_path="dice_val.png",
        y_min=y_min,
        y_max=y_max,
    )

    print("Hotovo ✅")
    print("Uloženo:")
    print(" - dice_train.png")
    print(" - dice_val.png")


if __name__ == "__main__":
    main()
