import torch
from Src.Utils.metrics import binary_dice, binary_jaccard, dice_per_channel, iou_per_channel
from monai.metrics import DiceMetric, compute_iou
from monai.inferers import sliding_window_inference
import numpy as np
import pandas as pd
import math
import torch.nn.functional as F

def testing_loop(model, dataset,files, check_EPS=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = DiceMetric(include_background=True, reduction="mean")
    model.to(device)
    model.eval()

    results = {"image_name": [], "dice": [], "iou": [], "monai_dice": [], "monai_iou": []}
    with torch.no_grad():
        idx = 0
        for inputs, labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
            r1_dice = binary_dice(outputs, labels) # use custom metric
            r2_dice = metric(outputs, labels).item() # use MONAI metric
            r1_iou = binary_jaccard(outputs, labels)
            r2_iou = compute_iou(y_pred=outputs, y=labels, include_background=False).item()
            if math.isnan(r2_dice) or math.isnan(r2_iou):
                print(
                    f"File: {files[idx]}, Dice: {r1_dice:.4f}, IOU: {r1_iou:.4f}, MONAI Dice: {r2_dice:.4f}, MONAI IOU: {r2_iou:.4f}")
            else:
                assert np.abs(
                    r1_dice - r2_dice) < check_EPS, "Custom metric and MONAI metric do not match! Check implementation or EPS value."
                assert np.abs(r1_iou - r2_iou) < check_EPS, "Custom IOU and MONAI IOU do not match! Check implementation or EPS value."

            results["image_name"].append(files[idx])
            results["dice"].append(r1_dice)
            results["iou"].append(r1_iou)
            results["monai_dice"].append(r2_dice)
            results["monai_iou"].append(r2_iou)
            idx+= 1
    return pd.DataFrame(results)



def testing_loop_multiclass(model, dataset,files, check_EPS=1e-3, classes=7):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_dice = DiceMetric(include_background=True, reduction="none")

    model.to(device)
    model.eval()

    results = {"image_name": [], "dice_mean": [], "iou_mean": []}
    for c in range(0,classes-1):
        results[f"dice_class_{c}"] = []
        results[f"iou_class_{c}"] = []

    with torch.no_grad():
        idx = 0
        for inputs, labels in dataset:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            #print(f"Output shape: {outputs.shape}, Labels shape: {labels.shape}")
            pred_classes = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            pred_onehot = F.one_hot(pred_classes, num_classes=classes).permute(0, 3, 1, 2).float()

            r1_dice_channels = dice_per_channel(pred_onehot[:,1:,:,:], labels[:,1:,:,:]) # nebrat background
            r1_dice_mean= np.nanmean(r1_dice_channels)
            # MONAI Dice per class
            metric_dice.reset()
            r2_dice_channels = metric_dice(pred_onehot, labels).tolist()[0][1:]  # Ignorovat background
            r2_dice_mean = np.nanmean(r2_dice_channels)
            #IOU per class

            r1_iou_channels = iou_per_channel(pred_onehot[:,1:,:,:], labels[:,1:,:,:])
            r1_iou_mean = np.nanmean(r1_iou_channels)
            r2_iou_channels = compute_iou(y_pred=pred_onehot[:,1:,:,:], y=labels[:,1:,:,:], include_background=True).tolist()[0][:]  # Ignorovat background
            r2_iou_mean= np.nanmean(r2_iou_channels)

            if math.isnan(r2_dice_mean) or math.isnan(r2_iou_mean):
                print(
                    f"File: {files[idx]}, Dice: {r1_dice_mean:.4f}, IOU: {r1_iou_mean:.4f}, MONAI Dice: {r2_dice_mean:.4f}, MONAI IOU: {r2_iou_mean:.4f}")
            # else:
            #    if np.abs(r1_dice_mean - r2_dice_mean) > check_EPS or np.abs(r1_iou_mean - r2_iou_mean) > check_EPS:
            #     print(f"File: {files[idx]}")
            #     print(f"r1_dice: {r1_dice_mean:.4f}\tmonai_dice: {r2_dice_mean:.4f}")
            #     print(f"Dice per channel: {r1_dice_channels}")
            #     print(f"monai per channel: {r2_dice_channels}")
            #     print(f"r1_iou: {r1_iou_mean:.4f}\tmonai_iou: {r2_iou_mean:.4f}")
            #     print(f"IOU per channel: {r1_iou_channels}")
            #     print(f"monai per channel: {r2_iou_channels}")
            #     print("*"*50)
            #     exit(-1)

            results["image_name"].append(files[idx])
            results["dice_mean"].append(r2_dice_mean)
            results["iou_mean"].append(r2_iou_mean)
            #results["monai_dice"].append(r2_dice_mean)
            #results["monai_iou"].append(r2_iou_mean)
            for c in range(0, classes-1):
                results[f"dice_class_{c}"].append(r2_dice_channels[c])
                results[f"iou_class_{c}"].append(r2_iou_channels[c])


            # print(f"File: {files[idx]}")
            # print(f"r1_dice: {r1_dice_mean:.4f}\tmonai_dice: {r2_dice_mean:.4f}")
            # print(f"Dice per channel: {r1_dice_channels}")
            # print(f"monai per channel: {r2_dice_channels}")
            # print(f"r1_iou: {r1_iou_mean:.4f}\tmonai_iou: {r2_iou_mean:.4f}")
            # print(f"IOU per channel: {r1_iou_channels}")
            # print(f"monai per channel: {r2_iou_channels}")
            # print("*"*50)


            idx+= 1

    return pd.DataFrame(results)

def testing_loop_sw_multiclass_to_df(
    model,
    dataloader,
    files,
    roi_size=(512, 512),
    sw_batch_size=4,
    overlap=0.25,
    classes=7,
    include_background=True,   # DiceMetric: True/False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dice_metric = DiceMetric(include_background=include_background, reduction="none")

    # Výstup jako v testing_loop_multiclass: per-image mean + per-class
    results = {"image_name": [], "dice_mean": [], "iou_mean": []}
    # Per-class uložíme bez backgroundu jako class_1..class_(C-1) (přehlednější)
    for c in range(1, classes):
        results[f"dice_class_{c}"] = []
        results[f"iou_class_{c}"] = []

    with torch.no_grad():
        idx = 0
        for batch in dataloader:
            # podporuje (x,y) i dict
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch["image"], batch["label"]

            x = x.to(device)  # (B,3,H,W)
            y = y.to(device)  # (B,7,H,W) one-hot

            # sliding window inference -> logits (B,7,H,W)
            logits = sliding_window_inference(
                inputs=x,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode="gaussian",
                device=device,
                sw_device=device,
            )

            # logits -> onehot
            pred_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)  # (B,H,W)
            pred_onehot = F.one_hot(pred_class, num_classes=classes).permute(0, 3, 1, 2).float()  # (B,7,H,W)

            # Dice per class (B,7)
            dice_metric.reset()
            dice_vals = dice_metric(pred_onehot, y)  # (B,7)
            dice_vals = dice_vals.detach().cpu().numpy()

            # IoU per class bez backgroundu (B, C-1)
            iou_vals = compute_iou(
                y_pred=pred_onehot,
                y=y,
                include_background=False
            ).detach().cpu().numpy()  # (B, classes-1)

            # U tebe bývá batch_size=1 při testu -> bereme [0]
            d = dice_vals[0]          # (7,)
            i = iou_vals[0]           # (6,)

            # mean bez backgroundu
            dice_mean = float(np.nanmean(d[1:]))
            iou_mean = float(np.nanmean(i))

            results["image_name"].append(files[idx])
            results["dice_mean"].append(dice_mean)
            results["iou_mean"].append(iou_mean)

            # per-class (1..C-1)
            for c in range(1, classes):
                results[f"dice_class_{c}"].append(float(d[c]))
                results[f"iou_class_{c}"].append(float(i[c-1]))

            idx += 1

    return pd.DataFrame(results)




