import os
import pandas as pd

def create_fold_info():
    return  {
       "train": {
         "image_name": [], "image_path": [],
         "label_name": [], "label_path": []
       },
       "val": {
           "image_name": [], "image_path": [],
           "label_name": [], "label_path": []
       }
     }

def fill_fold_info(fold_info, fold_type, image_name, label_name,images_path, labels_path):
    """
    Fills the fold information with image and label names and paths.
    :param fold_info:
    :param fold_type:
    :param image_name:
    :param label_name:
    :param images_path:
    :param labels_path:
    :return:
    """
    fold_info[fold_type]["image_name"].append(image_name)
    fold_info[fold_type]["label_name"].append(label_name)
    fold_info[fold_type]["image_path"].append(os.path.join(images_path, image_name))
    fold_info[fold_type]["label_path"].append(os.path.join(labels_path, label_name))
    if not os.path.exists(fold_info[fold_type]["image_path"][-1]):
        print(f"Image {fold_info[fold_type]['image_path'][-1]} does not exist.")

    if not os.path.exists(fold_info[fold_type]["label_path"][-1]):
        print(f"Label {fold_info[fold_type]['label_path'][-1]} does not exist.")



def get_split_files(dataset_name=None, images_path=None, labels_path=None,folds_path=None,k=5,max_files=1000_000):
   """
    Get the split files for the dataset.
   :param dataset_name:
   :param images_path:
   :param labels_path:
   :param folds_path:
   :param k:
   :return:
   """
   assert dataset_name is not None, "dataset_name must be provided"
   assert images_path is not None, "images_path must be provided"
   assert labels_path is not None, "labels_path must be provided"
   assert folds_path is not None, "folds_path must be provided"
   splits_files = []
   if max_files is None:
       max_files = 1000_000
   for i in range(k):
     train_fold_name = f"train_{dataset_name}_fold_{i}.csv"
     val_fold_name = f"val_{dataset_name}_fold_{i}.csv"
     train_fold_path = os.path.join(folds_path, train_fold_name)
     val_fold_path = os.path.join(folds_path, val_fold_name)
     if not os.path.exists(train_fold_path):
       print(os.getcwd())
       print(f"Train fold {train_fold_name} does not exist")
       continue
     df_train_fold = pd.read_csv(train_fold_path)
     df_val_fold = pd.read_csv(val_fold_path)
     fold_info = create_fold_info()
     for index, row in df_train_fold.iterrows():
       image_name = row["image"]
       label_name = row["label"]
       fill_fold_info(fold_info, "train", image_name, label_name, images_path, labels_path)
       if index >= max_files-1:
           break
     for index, row in df_val_fold.iterrows():
       image_name = row["image"]
       label_name = row["label"]
       fill_fold_info(fold_info, "val", image_name, label_name, images_path, labels_path)
       if index >= max_files-1:
           break

     splits_files.append(fold_info)
   return splits_files



if __name__ == "__main__":
    print("Data utils")
    data_path = "../../../Data-gauss-kaskada/Pets/Pets_converted"
    folds_path = f"{data_path}/Folds"
    images_path = os.path.join(data_path, "Images")
    labels_path = os.path.join(data_path, "MaskNPY")
    get_split_files(images_path=images_path,labels_path=labels_path, folds_path=folds_path, dataset_name="pets")