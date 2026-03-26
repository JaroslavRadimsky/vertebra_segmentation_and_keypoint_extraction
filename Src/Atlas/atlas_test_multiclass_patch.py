import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from monai.losses import DiceLoss
from monai.networks.nets import UNet
from Src.Models.training import load_model
from Src.Models.testing import testing_loop, testing_loop_multiclass
from Src.Models.testing import testing_loop_sw_multiclass_to_df
from Src.Utils.data_utils import get_split_files
from Src.Utils.replicability import set_seed, make_worker_seed_fn, get_generator
from Src.Atlas.atlas_dataset import AtlasDataset
from Src.Atlas.atlas_model import get_test_transformation, get_test_transformation_with_heqv # tuto transformaci budu pouzivat

import torch
from torch.utils.data import  DataLoader

#---------------general configuration-------------------
MAX_FILES = 10_000 #10_000 # pocet souboru pro testovani
START_FOLD = 0 # od ktereho foldu se zacne, 0 = vsechny foldy
config = {}
config["NAME"] = f"Atlas-heqv-multi-patch-{MAX_FILES}" # jmeno modelu heqv nebo bez
config["NUM_CLASSES"]=7 # 6 obratle + pozadí
config["BINARY"]=False # pokud je True, tak se upravi labely na 0,1 (0 = pozadí, 1 = objekt) v datasetu
config["MAX_EPOCHS"]=600 # pak asi 600
config["LR"] = 1e-2
config["TRAIN_BATCH_SIZE"] = 1 #32 #pro vypis vysledku za kazdy soubor
config["TEST_BATCH_SIZE"] =  1 #32
config["TRAIN"] = False
config["LOAD_TRAIN_MODEL"] = f"../../Models/{config['NAME']}"
config["SAVE_EPOCH"] = 50
config["MIN_VAL_DICE"] = 0.65
config["SEED"]= 42 # seed pro replicabilitu
config["HEQV"] = True # pouziti Histogram Equalization



if __name__=="__main__":
   # set seed pro replicabilitu
   set_seed(config["SEED"]) # nastaveni seed pro replicabilitu
   worker_init_fn = make_worker_seed_fn(config["SEED"])
   seeded_generator = get_generator(config["SEED"])
   #==========================
   print("Atlas segmentation model")
   data_path = "../../../data_Atlas-vertebra"
   folds_path = f"../../../Data-gauss-kaskada/Atlas_vertebra_folds"
   images_path = os.path.join(data_path, "datasets-PNG")
   labels_path = os.path.join(data_path, "datasets-NPY")
   folds= get_split_files(images_path=images_path,
                          labels_path=labels_path,
                          folds_path=folds_path,
                          dataset_name="atlas_vertebra", max_files=MAX_FILES) #TODO zmenit na MAX_FILES
   for fold_id, fold in enumerate(folds[START_FOLD:START_FOLD+1], start=START_FOLD):
      print(f"Fold {fold_id}")
  #   print(f"Train images: {len(fold['train']['image_name'])}")
  #    print(f"Train labels: {len(fold['train']['label_name'])}")
  #    print(f"Val images: {len(fold['val']['image_name'])}")
  #    print(f"Val labels: {len(fold['val']['label_name'])}")
      if config["HEQV"]:
         train_transformation = get_test_transformation_with_heqv()
         val_transformation = get_test_transformation_with_heqv()
      else:
         train_transformation = get_test_transformation()
         val_transformation = get_test_transformation()
      train_dataset=AtlasDataset(images=fold['train']['image_path'],
                                masks=fold['train']['label_path'],
                                classes=config["NUM_CLASSES"],
                                image_transform=train_transformation,
                                 binary=config["BINARY"],)
      train_loader = DataLoader(train_dataset,
                                batch_size=config["TRAIN_BATCH_SIZE"],
                                shuffle=False,
                                num_workers=1,
                                pin_memory=torch.cuda.is_available(),
                                worker_init_fn=worker_init_fn,
                                generator=seeded_generator
                                )

      test_dataset=AtlasDataset(images=fold['val']['image_path'],
                               masks=fold['val']['label_path'],
                               classes=config["NUM_CLASSES"],
                               image_transform=val_transformation,
                                binary=config["BINARY"],)
      test_loader = DataLoader(test_dataset,
                                 batch_size=config["TEST_BATCH_SIZE"],
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=torch.cuda.is_available(),
                                 worker_init_fn=worker_init_fn,
                                 generator=seeded_generator
                               )

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      model = UNet(
          spatial_dims=2,  # Pro 2D segmentaci
          in_channels=3,  # Počet kanálů vstupního obrázku (RGB = 3)
          out_channels=config["NUM_CLASSES"],  # Počet tříd
          channels=(16, 32, 64, 128, 256),  # Počet kanálů v jednotlivých vrstvách
          strides=(2, 2, 2, 2),  # Stride pro downsampling
          num_res_units=2  # Počet reziduálních bloků
      ).to(device)
      optimizer = torch.optim.Adam(model.parameters(), config["LR"])
      if config["LOAD_TRAIN_MODEL"] is not None:  # Pokud je zadáno jméno modelu, tak se načte
       # load stored model for testing
         model, optimizer, epoch = load_model(model, optimizer=optimizer,
                                                   filepath=f"{config['LOAD_TRAIN_MODEL']}-fold-{fold_id}-final.pth",
                                                   device=device)
         result_df_fold_val = testing_loop_sw_multiclass_to_df(
            model,
            test_loader,
            files=fold['val']['image_name'],
            roi_size=(512, 512),
            sw_batch_size=4,
            overlap=0.9,
            classes=config["NUM_CLASSES"],
         )

         result_df_fold_train = testing_loop_sw_multiclass_to_df(
            model,
            train_loader,
            files=fold['train']['image_name'],
            roi_size=(512, 512),
            sw_batch_size=4,
            overlap=0.9,
            classes=config["NUM_CLASSES"],
         )
         result_df_fold_val.to_csv(f"../../Results/Atlas/multiclass/{config['NAME']}-fold-{fold_id}-val-xxx.csv", index=False)
         result_df_fold_train.to_csv(f"../../Results/Atlas/multiclass/{config['NAME']}-fold-{fold_id}-train-xxx.csv", index=False)
         print(f"---------- fold {fold_id} --------")
         print("VALIDATION RESULTS")
         print(result_df_fold_val.describe().loc["mean"])
         print("TRAINING RESULTS")
         print(result_df_fold_train.describe().loc["mean"])





