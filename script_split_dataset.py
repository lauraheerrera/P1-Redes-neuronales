import os
import shutil
import random
import kagglehub

def clean_and_split_dataset(download_path, output_path, val_ratio=0.2, seed=42):
    """
    Reorganiza las carpetas duplicadas y divide seg_train en 80% train y 20% val.
    """

    random.seed(seed)

    # Detectar la carpeta correcta de seg_train
    if os.path.isdir(os.path.join(download_path, 'seg_train', 'seg_train')):
        train_src = os.path.join(download_path, 'seg_train', 'seg_train')
    else:
        train_src = os.path.join(download_path, 'seg_train')

    if os.path.isdir(os.path.join(download_path, 'seg_test', 'seg_test')):
        test_src = os.path.join(download_path, 'seg_test', 'seg_test')
    else:
        test_src = os.path.join(download_path, 'seg_test')

    # Crear carpetas de salida
    train_out = os.path.join(output_path, 'train')
    val_out   = os.path.join(output_path, 'val')
    test_out  = os.path.join(output_path, 'test')

    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    # -----------------------------
    # Copiar test tal cual
    # -----------------------------
    for cls in os.listdir(test_src):
        cls_path = os.path.join(test_src, cls)
        if os.path.isdir(cls_path):
            os.makedirs(os.path.join(test_out, cls), exist_ok=True)
            for f in os.listdir(cls_path):
                shutil.copy2(os.path.join(cls_path, f), os.path.join(test_out, cls, f))

    # -----------------------------
    # Dividir seg_train en train/val
    # -----------------------------
    for cls in os.listdir(train_src):
        cls_path = os.path.join(train_src, cls)
        if os.path.isdir(cls_path):
            images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
            random.shuffle(images)
            val_count = int(len(images) * val_ratio)
            val_images = images[:val_count]
            train_images = images[val_count:]

            os.makedirs(os.path.join(train_out, cls), exist_ok=True)
            os.makedirs(os.path.join(val_out, cls), exist_ok=True)

            for f in train_images:
                shutil.copy2(os.path.join(cls_path, f), os.path.join(train_out, cls, f))
            for f in val_images:
                shutil.copy2(os.path.join(cls_path, f), os.path.join(val_out, cls, f))

    print(f"Dataset reorganizado en {output_path}")
    print(f"Train/Val/Test listos con 80/20/total del test original")

if __name__ == "__main__":
    download_path = kagglehub.dataset_download("puneet6060/intel-image-classification")
    output_path = r"\Dataset"
    clean_and_split_dataset(download_path, output_path, val_ratio=0.2)