import torch.nn as nn
import torch
"""
下面这个类很重要，可以自动管理在多卡之间的一个具体数值变量的reduce（显卡之间归并数据）
可以用于实现算法的参考。
"""
# from training.utils.misc import MetricLogger

"""
改变这个接口的主要目的是image-level的指标和pixel-level的指标的计算方式不同
"""
class AbstractEvaluator(object): 
    def __init__(self) -> None:
        self.name = None
        self.desc = None
        self.threshold = None
    def batch_update(self, predict, pred_label, mask, shape_mask=None, *args, **kwargs):
        """
        本函数在每个batch结尾update。
        """
        raise NotImplementedError
    def epoch_update(self):
        """
        理论上这个时候没有新的数据了，所以没有输入参数。
        
        功能：在显卡之间收集所有在整个epoch内统计的指标，然后返回最终期望的信息。
        """
        raise NotImplementedError
    def recovery(self):
        raise NotImplementedError
    

class PixelF1(AbstractEvaluator):
    def __init__(self, threshold = 0.5, mode = "origin") -> None:
        super().__init__()
        self.name = "pixel-level F1"
        self.desc = "pixel-level F1"
        self.threshold = threshold
        self.image_num = 0
        #  mode : "origin, reverse, double"
        self.mode = mode

    def Cal_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        mask = (mask > threshold).float()
        if(shape_mask != None):
            TP = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum(predict * mask, dim=(1, 2, 3))  
            TN = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3)) 
            FP = torch.sum(predict * (1-mask), dim=(1, 2, 3)) 
            FN = torch.sum((1-predict) * mask, dim=(1, 2, 3))         
        return TP, TN, FP, FN

    def Cal_Reverse_Confusion_Matrix(self, predict, mask, shape_mask):
        """compute local confusion matrix for a batch of predict and target masks
        Args:
            predict (_type_): _description_
            mask (_type_): _description_
            region (_type_): _description_
            
        Returns:
            TP, TN, FP, FN
        """
        threshold = self.threshold
        predict = (predict > threshold).float()
        if(shape_mask != None):
            TP = torch.sum((1-predict) * mask * shape_mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask) * shape_mask, dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask) * shape_mask, dim=(1, 2, 3))
            FN = torch.sum(predict * mask * shape_mask, dim=(1, 2, 3))
        else:
            TP = torch.sum((1-predict) * mask, dim=(1, 2, 3))
            TN = torch.sum(predict * (1-mask), dim=(1, 2, 3))
            FP = torch.sum((1-predict) * (1-mask), dim=(1, 2, 3))
            FN = torch.sum(predict * mask, dim=(1, 2, 3))
        return TP, TN, FP, FN

    def Cal_F1(self, TP, TN, FP, FN):
        """_summary_

        Args:
            TP (_type_): _description_
            TN (_type_): _description_
            FP (_type_): _description_
            FN (_type_): _description_

        Returns:
            _type_: _description_
        """
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)
        # F1 = torch.mean(F1) # fuse the Batch dimension
        return F1

    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs): # 注意这里只有pixel-level需要的信息
        if self.mode == "origin":
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "reverse":
            TP, TN, FP, FN = self.Cal_Reverse_Confusion_Matrix(predict, mask, shape_mask)
            F1 = self.Cal_F1(TP, TN, FP, FN)
        elif self.mode == "double":
            # todo
            TP, TN, FP, FN = self.Cal_Confusion_Matrix(predict, mask, shape_mask)
            F1 = torch.max(self.Cal_F1(TP, TN, FP, FN), self.Cal_F1(FN, FP, TN, TP))
        else:
            raise RuntimeError(f"Cal_F1 no mode name {self.mode}")
        
        return F1
    def epoch_update(self):
        return None

    def recovery(self):
        self.image_num = 0

