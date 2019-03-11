import numpy as np
import math

# 计算IOU
def cal_IoU(cy1, h1, cy2, h2):
    y_top1, y_bottom1 = cal_y(cy1, h1)
    y_top2, y_bottom2 = cal_y(cy2, h2)
    offset = min(y_top1, y_top2)
    y_top1 = y_top1 - offset
    y_top2 = y_top2 - offset
    y_bottom1 = y_bottom1 - offset
    y_bottom2 = y_bottom2 - offset
    line = np.zeros(max(y_bottom1, y_bottom2) + 1)
    for i in range(y_top1, y_bottom1 + 1):
        line[i] += 1
    for j in range(y_top2, y_bottom2 + 1):
        line[j] += 1
    union = np.count_nonzero(line, 0)
    intersection = line[line == 2].size
    return float(intersection)/float(union)


# 通过h和cy计算anchor的上下界限
def cal_y(cy, h):
    y_top = int(cy - (float(h) - 1) / 2.0)
    y_bottom = int(cy + (float(h) - 1) / 2.0)
    return y_top, y_bottom


def valid_anchor(cy, h, height):
    top, bottom = cal_y(cy, h)
    if top < 0:
        return False
    if bottom > (height * 16 - 1):  # CTPN中锚框的宽高相等均为16，一张图片被划分成了N个16*16锚框
        return False
    return True


# 输入一个实际的bbox，其对应的实际的anchor，网络的预测得分score
# 求一个bbox中的正、负样本，位置预测、边缘优化值
def tag_anchor(gt_anchor, cnn_output, gt_box):
    # from 11 to 273, divide 0.7 each time
    # 0.7和IOU阈值是一样的，这样能保证在某个尺寸满足0.7时，下个尺寸不再满足，确保anchor唯一性
    anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]  
    # whole image h and w
    height = cnn_output.shape[2]  # 这是图片的高度么？
    width = cnn_output.shape[3]  # 这是图片的宽度么？
    positive = []
    negative = []
    vertical_reg = []
    side_refinement_reg = []
    x_left_side = min(gt_box[0], gt_box[6])  # 整个文本框bbox的最左侧
    x_right_side = max(gt_box[2], gt_box[4])  # 整个文本框bbox的最右侧
    left_side = False
    right_side = False

    # 遍历一个ground truth box中的所有anchors，是一个个小anchor
    for a in gt_anchor:

        # a[0]表示anchor的水平id，如果水平id比图片宽度还宽，跳过
        if a[0] >= int(width - 1):
            continue

        # 判断这个anchor的左边界是否为整个bbox的左边界
        if x_left_side in range(a[0] * 16, (a[0] + 1) * 16):
            left_side = True
        else:
            left_side = False

        # 判断这个anchor的右边界是否为整个bbox的右边界
        if x_right_side in range(a[0] * 16, (a[0] + 1) * 16):
            right_side = True
        else:
            right_side = False

        # 针对这个小anchor
        iou = np.zeros((height, len(anchor_height)))
        temp_positive = []
        for i in range(iou.shape[0]):  # 默认锚框中的哪个y和h能和gt_anchor的y和h产生大于0.7的IOU
            for j in range(iou.shape[1]):
                if not valid_anchor((float(i) * 16.0 + 7.5), anchor_height[j], height):
                    continue
                iou[i][j] = cal_IoU((float(i) * 16.0 + 7.5), anchor_height[j], a[1], a[2])

                if iou[i][j] > 0.7:
                    temp_positive.append((a[0], i, j, iou[i][j]))
                    if left_side:
                        o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))
                    if right_side:
                        o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                        side_refinement_reg.append((a[0], i, j, o))

                if iou[i][j] < 0.5:
                    negative.append((a[0], i, j, iou[i][j]))

                if iou[i][j] > 0.5:
                    vc = (a[1] - (float(i) * 16.0 + 7.5)) / float(anchor_height[j])
                    vh = math.log10(float(a[2]) / float(anchor_height[j]))
                    vertical_reg.append((a[0], i, j, vc, vh, iou[i][j]))

        if len(temp_positive) == 0:
            max_position = np.where(iou == np.max(iou))
            temp_positive.append((a[0], max_position[0][0], max_position[1][0], np.max(iou)))

            if left_side:
                o = (float(x_left_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))
            if right_side:
                o = (float(x_right_side) - (float(a[0]) * 16.0 + 7.5)) / 16.0
                side_refinement_reg.append((a[0], max_position[0][0], max_position[1][0], o))

            if np.max(iou) <= 0.5:
                vc = (a[1] - (float(max_position[0][0]) * 16.0 + 7.5)) / float(anchor_height[max_position[1][0]])
                vh = math.log10(float(a[2]) / float(anchor_height[max_position[1][0]]))
                vertical_reg.append((a[0], max_position[0][0], max_position[1][0], vc, vh, np.max(iou)))
        positive += temp_positive
    return positive, negative, vertical_reg, side_refinement_reg
