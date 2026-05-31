import os
import cv2
import numpy as np
from scipy.io import loadmat
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATA_ROOT = os.environ.get("CRCDANET_DATA_ROOT", "./datasets")
RESULT_ROOT = os.environ.get("CRCDANET_RESULT_ROOT", "./z_result")


def _path(*parts):
    return os.path.join(*parts)


def _read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype("float32")


def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype("float32")


def _loadmat_key(path, key):
    data = loadmat(path)
    if key not in data:
        raise KeyError(f"Key '{key}' was not found in {path}. Available keys: {list(data.keys())}")
    return data[key].astype("float32")


def _read_pred_mat(path):
    return _loadmat_key(path, "pred")


def ensure_3d(array):
    if array.ndim == 2:
        return array[:, :, None]
    return array


def ensure_3ch_if_1ch(array):
    array = ensure_3d(array)
    if array.shape[-1] == 1:
        return np.repeat(array, 3, axis=-1)
    return array


def read_mat_key(data, keys):
    for key in keys:
        if key in data:
            return data[key].astype("float32")
    raise KeyError(f"None of the keys {keys} was found. Available keys: {list(data.keys())}")


def _loop_ratio(loop):
    ratio_map = {0: "0.01", 1: "0.01", 2: "0.01", 3: "0.01", 4: "0.01", 5: "0.01"}
    return ratio_map.get(loop, "0.01")


def _cd_pred_path(pair_name, loop):
    if loop == 0:
        return None
    return _path(RESULT_ROOT, pair_name, "result", "CD_result", f"CD_{loop - 1}_{_loop_ratio(loop)}", "predict_mat.mat")


def get_Islandtown_dataset():
    rgb_path = _path(DATA_ROOT, "Islandtown", "Img11-Ac.ppm")
    sar_path = _path(DATA_ROOT, "Islandtown", "Img11-B.ppm")
    gt_path = _path(DATA_ROOT, "Islandtown", "Img11-C.ppm")

    img_t1 = _read_rgb(sar_path)
    img_t2 = _read_rgb(rgb_path)
    img_gt = (_read_gray(gt_path) / 255).astype("float32")
    return img_t1, img_t2, img_gt


def get_shuguang_dataset():
    t1_path = _path(DATA_ROOT, "shuguang", "shuguang_1.bmp")
    t2_path = _path(DATA_ROOT, "shuguang", "shuguang_2.bmp")
    gt_path = _path(DATA_ROOT, "shuguang", "shuguang_gt.bmp")

    img_t1 = _read_rgb(t1_path)
    img_t2 = _read_rgb(t2_path)
    img_gt = (_read_gray(gt_path) / 255).astype("float32")
    return img_t1, img_t2, img_gt


def get_Islandtown_loop(loop):
    img_t1, img_t2, gt = get_Islandtown_dataset()
    pred_path = _cd_pred_path("Islandtown_shuguang", loop)
    if pred_path is not None and os.path.exists(pred_path):
        gt = _read_pred_mat(pred_path)
    return img_t1, img_t2, gt


def get_shuguang_loop(loop):
    img_t1, img_t2, gt = get_shuguang_dataset()
    pred_path = _cd_pred_path("Islandtown_shuguang", loop)
    if pred_path is not None and os.path.exists(pred_path):
        gt = _read_pred_mat(pred_path)
    return img_t1, img_t2, gt


def _load_generated_pair(base_dir, prefix):
    rgb_t1 = _loadmat_key(_path(base_dir, "sar_rgb", f"{prefix}_T1_rgb.mat"), "T1")
    rgb_t2 = _loadmat_key(_path(base_dir, "sar_rgb", f"{prefix}_T1_rgb_original.mat"), "T1_original")
    sar_t1 = _loadmat_key(_path(base_dir, "rgb_sar", f"{prefix}_T2_sar.mat"), "T2")
    sar_t2 = _loadmat_key(_path(base_dir, "rgb_sar", f"{prefix}_T2_sar_original.mat"), "T2_original")
    return rgb_t1, rgb_t2, sar_t1, sar_t2


def _load_target_generated_pair(loop):
    base = _path(RESULT_ROOT, "Islandtown_shuguang", "result", "DF_result")
    rgb_dir = _path(base, f"sar_rgb_{loop}")
    sar_dir = _path(base, f"rgb_sar_{loop}")

    rgb_t1 = _loadmat_key(_path(rgb_dir, f"shuguang_domain_RGB_ddpm_{loop}.mat"), "T1")
    rgb_t2 = _loadmat_key(_path(rgb_dir, f"shuguang_domain_RGB_ddpm_original_{loop}.mat"), "T1_original")
    sar_t1 = _loadmat_key(_path(sar_dir, f"shuguang_domain_SAR_ddpm_{loop}.mat"), "T2")
    sar_t2 = _loadmat_key(_path(sar_dir, f"shuguang_domain_SAR_ddpm_original_{loop}.mat"), "T2_original")
    return rgb_t1, rgb_t2, sar_t1, sar_t2


def get_cd_Islandtown_loop(loop):
    base = _path(RESULT_ROOT, "source", "Islandtown", "result", "DF_result")
    img_rgb_t1, img_rgb_t2, img_sar_t1, img_sar_t2 = _load_generated_pair(base, "Islandtown")
    _, _, img_gt0 = get_Islandtown_dataset()
    img_gt = img_gt0.copy()
    return img_rgb_t1, img_rgb_t2, img_sar_t1, img_sar_t2, img_gt, img_gt0


def get_cd_shuguang_loop(loop):
    img_rgb_t1, img_rgb_t2, img_sar_t1, img_sar_t2 = _load_target_generated_pair(loop)
    _, _, img_gt0 = get_shuguang_dataset()

    pred_path = _cd_pred_path("Islandtown_shuguang", loop)
    if pred_path is not None and os.path.exists(pred_path):
        img_gt = _read_pred_mat(pred_path)
    else:
        img_gt = img_gt0.copy()

    return img_rgb_t1, img_rgb_t2, img_sar_t1, img_sar_t2, img_gt, img_gt0


def get_cd_test_shuguang_loop(loop):
    img_rgb_t1, img_rgb_t2, img_sar_t1, img_sar_t2 = _load_target_generated_pair(loop)
    _, _, img_gt = get_shuguang_dataset()
    return img_rgb_t1, img_rgb_t2, img_sar_t1, img_sar_t2, img_gt


def get_dataset(dataset_name):
    if dataset_name == "Islandtown":
        return get_Islandtown_dataset()
    if dataset_name == "Shuguang":
        return get_shuguang_dataset()
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_dataset_loop(dataset_name, loop):
    if dataset_name == "Islandtown_domain":
        return get_Islandtown_loop(loop)
    if dataset_name == "shuguang_domain":
        return get_shuguang_loop(loop)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_cd_dataset_loop(dataset_name, loop):
    if dataset_name == "Islandtown_domain":
        return get_cd_Islandtown_loop(loop)
    if dataset_name == "shuguang_domain":
        return get_cd_shuguang_loop(loop)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_dataset_CD_test(dataset_name, loop):
    if dataset_name == "shuguang_domain":
        return get_cd_test_shuguang_loop(loop)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")
