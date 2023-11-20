from sklearn import metrics
import numpy as np
import cv2

class Evaluation():
    # 计算各应用评价指标
    def __init__(self, label, pred):
        super(Evaluation, self).__init__()
        self.label = label / 255
        self.pred = pred / 255

    def ConfusionMatrix(self):
        raw = self.label.shape[0]
        col = self.label.shape[1]
        size = raw * col
        union = np.clip(((self.label + self.pred)), 0, 1)
        intersection = (self.label * self.pred)
        TP = int(intersection.sum())
        TN = int(size - union.sum())
        FP = int((self.pred - intersection).sum())
        FN = int((self.label - intersection).sum())

        c_num_or = int(union.sum())
        uc_num_or = int(size - intersection.sum())

        return TP, TN, FP, FN, c_num_or, uc_num_or

class Index():
    # 计算各应用评价指标
    def __init__(self, TPSum, TNSum, FPSum, FNSum, c_Sum_or, uc_Sum_or):
        super(Index, self).__init__()
        self.TP = TPSum
        self.TN = TNSum
        self.FP = FPSum
        self.FN = FNSum
        self.c_num_and = TPSum
        self.c_num_or = c_Sum_or
        self.uc_num_and = TNSum
        self.uc_num_or = uc_Sum_or

    def CD_indicators(self):


        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN

        FA = FP / (FP + TN)
        MA = FN / (FN + TP)
        TE = (FP + FN) / (TP + TN + FP + FN)

        return FA*100, MA*100, TE*100

    def Classification_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN


        OA = (TP + TN) / (TP + TN + FP + FN)
        kappa = metrics.cohen_kappa_score(label.flatten(), pred.flatten())
        AA = (TP / (TP + FN) + TN / (TN + FP)) / 2

        return OA*100, kappa*100, AA*100

    def Landsilde_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN


        Completeness = TP / (TP + FN)
        Correctness = TP / (TP + FP)
        Quality = TP / (TP + FP + FN)

        return Completeness*100, Correctness*100, Quality*100

    # return iou
    def IOU_indicator(self):
        c_num_and = self.c_num_and
        c_num_or = self.c_num_or
        uc_num_and = self.uc_num_and
        uc_num_or = self.uc_num_or



        c_iou = (c_num_and / c_num_or) * 100
        uc_iou = (uc_num_and / uc_num_or) * 100
        mIoU = (c_iou + uc_iou) / 2

        return mIoU, c_iou, uc_iou
        """
        if c_num_or == 0:
            c_iou = 100
        else:
            c_iou = (c_num_and / c_num_or) * 100
        uc_iou = (uc_num_and / uc_num_or) * 100
        mIoU = (c_iou + uc_iou) / 2
        return mIoU, c_iou, uc_iou
        """


    # return Precision, Recall, F1
    def ObjectExtract_indicators(self):
        TP = self.TP
        TN = self.TN
        FP = self.FP
        FN = self.FN


        OA = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)

        return OA*100, Precision*100, Recall*100, F1*100

if __name__ == "__main__":
    pred_path = "E:/RuanJian2/DeepLearning/Model/1/AGSCNet/data/test/image"
    label_path = "E:/RuanJian2/DeepLearning/Model/1/AGSCNet/data/test/label"
    pred = cv2.imread(pred_path)
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    """
    Indicators = Evaluation(label, pred)
    OA, kappa, AA = Indicators.Classification_indicators()
    FA, MA, TE = Indicators.CD_indicators()
    CP, CR, AQ = Indicators.Landsilde_indicators()
    IOU = Indicators.IOU_indicator()
    Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
    print("(OA, KC, AA)", OA, kappa, AA)
    print("(FA, MA, TE)", FA, MA, TE)
    print("(CP, CR, AQ)", CP, CR, AQ)
    print("(IoU, Precision, Recall, F1-score)", IOU, Precision, Recall, F1)
    """