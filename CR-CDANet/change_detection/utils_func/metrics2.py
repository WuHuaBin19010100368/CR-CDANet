import numpy as np


def calculate_confusion_matrix(preds, labels):
    """
    计算二分类问题的混淆矩阵。
    """
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    TP = float(TP)
    FP = float(FP)
    FN = float(FN)
    TN = float(TN)
    return TP, FP, TN, FN


def calculate_OA(TP, FP, TN, FN):
    """
    计算二分类问题的OA（Overall Accuracy）。
    """
    return (TP + TN) / (TP + FP + TN + FN)


def calculate_Kappa(TP, FP, TN, FN):
    """
    计算二分类问题的Kappa。
    """
    total = TP + FP + TN + FN
    po = (TP + TN) / total
    pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / (total * total)
    return (po - pe) / (1 - pe)


def calculate_Pr(TP, FP):
    """
    计算二分类问题的Pr（Precision）。
    """
    return TP / (TP + FP)


def calculate_Re(TP, FN):
    """
    计算二分类问题的Re（Recall）。
    """
    return TP / (TP + FN)


def calculate_F1(Pr, Re):
    """
    计算二分类问题的F1 Score。
    """
    return 2 * Pr * Re / (Pr + Re)

def calculate_AA(TP, FP, TN, FN):
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0  # 正类的准确率
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0  # 负类的准确率
    return (TPR + TNR) / 2

def calculate_metrics(preds, labels):
    """
    计算二分类问题的所有指标。
    """
    # if np.any((labels == 0) | (labels == 1)):
    TP, FP, TN, FN = calculate_confusion_matrix(preds, labels)

    OA = calculate_OA(TP, FP, TN, FN)
    print("TP:{} FP:{} TN:{} FN:{}".format(TP,FP,TN,FN))
    Kappa = calculate_Kappa(TP, FP, TN, FN)
    Pr = calculate_Pr(TP, FP)
    Re = calculate_Re(TP, FN)
    F1 = calculate_F1(Pr, Re)
    AA = calculate_AA(TP, FP, TN, FN)

    # return OA, Kappa, F1, Pr, Re, AA
    return OA, Kappa, F1, Pr, Re