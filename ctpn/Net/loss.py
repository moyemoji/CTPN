import torch.nn as nn
import random
import torch


class CTPN_Loss(nn.Module):
    """
    CTPN的LOSS分为三部分：
    ——h,y的regression loss，用的是SmoothL1Loss；
    ——score的classification loss，用的是CrossEntropyLoss；
    ——side refinement loss，用的是用的是SmoothL1Loss。
    """
    def __init__(self, using_cuda=False):
        super(CTPN_Loss, self).__init__()
        self.Ns = 128
        self.ratio = 0.5
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.Ls_cls = nn.CrossEntropyLoss()  # score的classification loss
        self.Lv_reg = nn.SmoothL1Loss()  # h,y的regression loss
        self.Lo_reg = nn.SmoothL1Loss()  # side refinement loss
        self.using_cuda = using_cuda

    def forward(self, score, vertical_pred, side_refinement, positive, negative, vertical_reg, side_refinement_reg):
        """
        :param score: prediction score 预测的分数
        :param vertical_pred: prediction vertical coordinate  预测的垂直坐标
        :param side_refinement: prediction side refinement  预测的边缘优化值
        :param positive: ground truth positive fine-scale box  实际的正样本？  实际anchor最适尺寸的默认锚框？
        :param negative: ground truth negative fine-scale box  实际的负样本？
        :param vertical_reg: ground truth vertical regression  实际的垂直回归？
        :param side_refinement_reg: ground truth side-refinement regression  实际的边缘优化？
        :return: total loss
        """
        # calculate classification loss
        positive_num = min(int(self.Ns * self.ratio), len(positive))
        negative_num = self.Ns - positive_num
        positive_batch = random.sample(positive, positive_num)
        negative_batch = random.sample(negative, negative_num)
        cls_loss = 0.0
        if self.using_cuda:
            for p in positive_batch:
                cls_loss += self.Ls_cls(score[0, p[2] * 2: ((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0),
                                        torch.LongTensor([1]).cuda())
            for n in negative_batch:
                cls_loss += self.Ls_cls(score[0, n[2] * 2: ((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0),
                                        torch.LongTensor([0]).cuda())
        else:
            for p in positive_batch:
                cls_loss += self.Ls_cls(score[0, p[2] * 2: ((p[2] + 1) * 2), p[1], p[0]].unsqueeze(0),
                                        torch.LongTensor([1]))
            for n in negative_batch:
                cls_loss += self.Ls_cls(score[0, n[2] * 2: ((n[2] + 1) * 2), n[1], n[0]].unsqueeze(0),
                                        torch.LongTensor([0]))
        cls_loss = cls_loss / self.Ns

        # calculate vertical coordinate regression loss
        v_reg_loss = 0.0
        Nv = len(vertical_reg)
        if self.using_cuda:
            for v in vertical_reg:
                v_reg_loss += self.Lv_reg(vertical_pred[0, v[2] * 2: ((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0),
                                          torch.FloatTensor([v[3], v[4]]).unsqueeze(0).cuda())
        else:
            for v in vertical_reg:
                v_reg_loss += self.Lv_reg(vertical_pred[0, v[2] * 2: ((v[2] + 1) * 2), v[1], v[0]].unsqueeze(0),
                                          torch.FloatTensor([v[3], v[4]]).unsqueeze(0))
        v_reg_loss = v_reg_loss / float(Nv)

        # calculate side refinement regression loss
        o_reg_loss = 0.0
        No = len(side_refinement_reg)
        if self.using_cuda:
            for s in side_refinement_reg:
                o_reg_loss += self.Lo_reg(side_refinement[0, s[2]: s[2] + 1, s[1], s[0]].unsqueeze(0),
                                          torch.FloatTensor([s[3]]).unsqueeze(0).cuda())
        else:
            for s in side_refinement_reg:
                o_reg_loss += self.Lo_reg(side_refinement[0, s[2]: s[2] + 1, s[1], s[0]].unsqueeze(0),
                                          torch.FloatTensor([s[3]]).unsqueeze(0))
        o_reg_loss = o_reg_loss / float(No)

        loss = cls_loss + v_reg_loss * self.lambda1 + o_reg_loss * self.lambda2
        return loss, cls_loss, v_reg_loss, o_reg_loss
