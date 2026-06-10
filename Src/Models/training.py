from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from monai.metrics import DiceMetric, MeanIoU
from Src.project_paths import RUNS_DIR, SEGMENTATION_CHECKPOINTS_DIR


def save_model(model, optimizer, epoch, filepath=SEGMENTATION_CHECKPOINTS_DIR / "model.pth"):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print(f"[saved] {filepath}")

def load_model(model, optimizer, filepath="model.pth", device="cuda"):
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    #print(f"[loaded] {filepath}, last epoch: {epoch}")
    return model, optimizer, epoch

def training_loop(model, loss_function, optimizer, train_loader, val_loader, config, lrconfig, fold_id, start_epoch=0):
    min_val_dice=config["MIN_VAL_DICE"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(RUNS_DIR / f'{config["NAME"]}-fold-{fold_id}')
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    if lrconfig["LRReduceOnPlato"]:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lrconfig["LR_RATIO"], patience=lrconfig["LR_PATIENCE"])
    else:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=lrconfig["T0"], T_mult=lrconfig["T_MULT"], eta_min=lrconfig["ETA_MIN"])

    best_val_dice = 0.0
    for epoch in range(start_epoch, config["MAX_EPOCHS"]):
        # trénovaní přes batche
        model.train()
        epoch_loss = 0
        total_batches = 0
        for batch_data in train_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            # print(f"Batch input data shape: {inputs.shape}")
            # print(f"Batch label data shape: {labels.shape}")
            # # krok optimizeru nad batchem
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"Model shape output: {outputs.shape}")
            loss = loss_function(outputs, labels)
            epoch_loss += loss.item()
            # Výpočet Dice koeficientu
            # Musí se převést výstup modelu na predikce (argmax)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)

            # print("-----")
            # print(outputs[0,:3,:3])

            dice_metric(outputs, labels)  # Aktualizace Dice metriky
            loss.backward()
            optimizer.step()
            total_batches += 1

        train_epoch_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        epoch_loss /= total_batches
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Dice/train", train_epoch_dice, epoch)
        writer.add_scalar("LR", current_lr, epoch)
        if epoch % config["SAVE_EPOCH"] == 0:
            save_model(model, optimizer, epoch, filepath=SEGMENTATION_CHECKPOINTS_DIR / f"{config['NAME']}-epoch-{epoch}-fold-{fold_id}.pth")

        # Validace
        model.eval()
        epoch_val_dice = 0.0
        total_val_batches = 0

        with torch.no_grad():  # Neprovádí zpětnou propagaci
            for inputs, labels in val_loader:  # val_loader je validační data
                # Převod na zařízení (GPU, CPU)
                inputs, labels = inputs.to(device), labels.to(device)
                # Predikce modelu
                outputs = model(inputs)
                # Výpočet Dice koeficientu
                # Musí se převést výstup modelu na predikce (argmax)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                dice_metric(outputs, labels)  # Aktualizace Dice metriky
                total_val_batches += 1
        # Sčítání metrik pro validační část
        epoch_val_dice += dice_metric.aggregate().item()
        if epoch_val_dice > best_val_dice and epoch_val_dice > min_val_dice :
            best_val_dice = epoch_val_dice
            save_model(model, optimizer, epoch, filepath=SEGMENTATION_CHECKPOINTS_DIR / f"{config['NAME']}-fold-{fold_id}-best.pth")
        writer.add_scalar("Dice/val", epoch_val_dice, epoch)
        dice_metric.reset()
        print(f"Epoch {epoch}: Loss: {epoch_loss:.4f}, Train DICE: {train_epoch_dice:.6f}, Val DICE: {epoch_val_dice:.6f}, LR: {current_lr:.6f}")
    writer.close()
