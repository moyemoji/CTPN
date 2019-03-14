import sys
sys.path.append('/home/sambauser/ctpn_batch_num/ctpn/lib')
import math
import copy
import cv2
import time
import os
import draw_image


# 从bbox生成anchor，把照片中的文本box转为小的条状anchor，ground truth
def generate_gt_anchor(img, box, anchor_width=16, draw_img_gt=None):
    """
    calsulate ground truth fine-scale box
    :param img: input image
    :param box: ground truth box (4 point)
    :param anchor_width:
    :return: tuple (position, h, cy)
    """
    if not isinstance(box[0], float):
        box = [float(box[i]) for i in range(len(box))]

    result = []
    # box数据格式为[x1 y1 x2 y2 x3 y3 x4 y4]，左上、左下、右上、右下
    left_anchor_num = int(math.floor(max(min(box[0], box[6]), 0) / anchor_width))  # the left side anchor of the text box, downwards
    right_anchor_num = int(math.ceil(min(max(box[2], box[4]), img.shape[1]) / anchor_width))  # the right side anchor of the text box, upwards

    # handle extreme case, the right side anchor may exceed the image width
    if right_anchor_num * 16 + 15 > img.shape[1]:
        right_anchor_num -= 1

    # combine the left-side and the right-side x_coordinate of a text anchor into one pair
    # 把一个box中的每个anchor左右边界以元组形式保存在position_pair中
    position_pair = [(i * anchor_width, (i + 1) * anchor_width - 1) for i in range(left_anchor_num, right_anchor_num)]

    # 计算每个gt anchor的真实位置，其实就是求解gt anchor的上边界和下边界
    y_top, y_bottom = cal_y_top_and_bottom2(img, position_pair, box)
    # y_top2, y_bottom2 = cal_y_top_and_bottom2(img, position_pair, box)
    # for i in range(len(y_top)):
    #     if abs(y_top[i] - y_top2[i]) > 5 or abs(y_bottom[i] - y_bottom2[i]) > 5:
    #         print("**********")
    #         print(y_top[i],y_bottom[i],y_top2[i],y_bottom2[i])
    # print()

    #print("image shape: %s, pair_num: %s, top_num:%s, bot_num:%s" % (img.shape, len(position_pair), len(y_top), len(y_bottom)))

    # 最后将每个anchor的位置(水平ID，从左到右第几个anchor)、anchor中心y坐标、anchor高度存储并返回
    for i in range(len(position_pair)):
        position = int(position_pair[i][0] / anchor_width)  # the index of anchor box
        h = y_bottom[i] - y_top[i] + 1  # the height of anchor box
        cy = (float(y_bottom[i]) + float(y_top[i])) / 2.0  # the center point of anchor box
        result.append((position, cy, h))  # result保存图片中box的所有ground truth anchor的水平位置、y、h
        draw_img_gt = draw_image.draw_box_h_and_c(draw_img_gt, position, cy, h)  # 把anchor画出来
    draw_img_gt = draw_image.draw_box_4pt(draw_img_gt, box, color=(0, 0, 255), thickness=1)
    return result, draw_img_gt


# cal the gt anchor box's bottom and top coordinate
# 计算anchor的上下边界，可以用来计算anchor的高h、纵坐标中心cy
def cal_y_top_and_bottom(raw_img, position_pair, box):
    """
    :param raw_img:
    :param position_pair: for example:[(0, 15), (16, 31), ...]
    :param box: gt box (4 point)
    :return: top and bottom coordinates for y-axis
    """
    img = copy.deepcopy(raw_img)
    y_top = []
    y_bottom = []
    height = img.shape[0]
    img[:, :, 0] = 0
    # for i in range(img.shape[0]):
    #     for j in range(img.shape[1]):
    #         img[i, j, 0] = 0
    top_flag = False
    bottom_flag = False
    img = draw_image.draw_box_4pt(img, box, color=(255, 0, 0))
    # calc top y coordinate, pixel from top to down loop
    for k in range(len(position_pair)):
        # calc top y coordinate
        for y in range(0, height-1):
            # loop each anchor, from left to right
            for x in range(position_pair[k][0], position_pair[k][1] + 1):
                if img[y, x, 0] == 255:
                    y_top.append(y)
                    top_flag = True
                    break 
            if top_flag is True:
                break
        # calc bottom y coordinate, pixel from down to top loop
        for y in range(height - 1, -1, -1):
            # loop each anchor, from left to right
            for x in range(position_pair[k][0], position_pair[k][1] + 1):
                if img[y, x, 0] == 255:
                    y_bottom.append(y)
                    bottom_flag = True
                    break
            if bottom_flag is True:
                break
        top_flag = False
        bottom_flag = False
    # print(y_top)
    # print(y_bottom)
    return y_top, y_bottom

# 将可能随机排列的坐标点，统一重新排列成左下、左上、右上、右下排列
def sortCoords(box):
    coords = [[box[0],box[1]],[box[2],box[3]],[box[4],box[5]],[box[6],box[7]]]
    coords_x = [box[0],box[2],box[4],box[6]]
    coords_x.sort()
    coords_left = []
    coords_right = []
    for i in range(4):
        if coords[i][0] == coords_x[0] or coords[i][0] == coords_x[1]:
            coords_left.append(coords[i])
        else:
            coords_right.append(coords[i])
    new_box = []
    if coords_left[0][1] < coords_left[1][1]:
        new_box += coords_left[0]
        new_box += coords_left[1]
    else:
        new_box += coords_left[1]
        new_box += coords_left[0]
    if coords_right[0][1] > coords_right[1][1]:
        new_box += coords_right[0]
        new_box += coords_right[1]
    else:
        new_box += coords_right[1]
        new_box += coords_right[0]

    return new_box

def calcY(x, param):
    a = param[0]
    b = param[1]
    c = param[2]
    if b!=0:
        return int(-(c+a*x)/b)
    else:
        return 0

def calcLine(box): 
    x1 = box[0]
    x2 = box[2]
    x3 = box[4]
    x4 = box[6]
    y1 = box[1]
    y2 = box[3]
    y3 = box[5]
    y4 = box[7]

    # l12
    a12 = y2 - y1
    b12 = x1 - x2
    c12 = x2 * y1 - x1 * y2
    line12 = [a12,b12,c12]

    # l34
    a34 = y4 - y3
    b34 = x3 - x4
    c34 = x4 * y3 - x3 * y4
    line34 = [a34,b34,c34]

    # l14
    a14 = y4 - y1
    b14 = x1 - x4
    c14 = x4 * y1 - x1 * y4
    line14 = [a14,b14,c14]

    # l23
    a23 = y3 - y2
    b23 = x2 - x3
    c23 = x3 * y2 - x2 * y3
    line23 = [a23,b23,c23]

    return [line14,line23,line12,line34]



def cal_y_top_and_bottom2(raw_img, position_pair, box):
    y_top = []
    y_bottom = []
    box = sortCoords(box)
    lines = calcLine(box)
    whichline = []
    for k in range(len(position_pair)):
        box_x = [box[0],box[2],box[4],box[6]]
        box_x.sort()
        box_left_left = box_x[0]
        box_left_right = box_x[1]
        box_right_left = box_x[2]
        box_right_right = box_x[3]
        anchor_middle_x = position_pair[k][0] + 7.5

        if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
            anchor_y_top = calcY(anchor_middle_x, lines[0])
            anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(1)
            continue

        elif anchor_middle_x > box_left_left and anchor_middle_x < box_left_right:  # 位于左边界上的
            if lines[2][1] == 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                continue
            k_l12 = -(lines[2][0]/lines[2][1])
            anchor_y_top = calcY(anchor_middle_x, lines[0])
            anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            if k_l12 > 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[2])
            else:
                anchor_y_top = calcY(anchor_middle_x, lines[2])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(2)
            continue

        elif anchor_middle_x <= box_left_left:
            anchor_middle_x = position_pair[k][1]  # 如果左边界外的，则用anchor右边缘替代中线
            if anchor_middle_x > box_right_right:  # 如果box很小时，anchor右边界超出box右边界，此时将anchor_middle_x替换为box_left_right
                anchor_middle_x = box_left_right
            if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(3)
                continue
            else:
                if lines[2][1] == 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                    y_top.append(anchor_y_top)
                    y_bottom.append(anchor_y_bottom)
                    continue
                k_l12 = -(lines[2][0]/lines[2][1])
                if k_l12 > 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[2])
                else:
                    anchor_y_top = calcY(anchor_middle_x, lines[2])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(4)
                continue

        elif anchor_middle_x > box_right_left and anchor_middle_x < box_right_right:  # 位于右边界上的
            if lines[3][1] == 0:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                continue
            k_l34 = -(lines[3][0]/lines[3][1])
            if k_l34> 0:
                anchor_y_top = calcY(anchor_middle_x, lines[3])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
            else:
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[3])
            y_top.append(anchor_y_top)
            y_bottom.append(anchor_y_bottom)
            whichline.append(5)
            continue

        elif anchor_middle_x >= box_right_right:
            anchor_middle_x = position_pair[k][0]  # anchor左边界替代中线
            if anchor_middle_x < box_left_left:  # 如果box很小时，anchor左边界超出box左边界，此时将anchor_middle_x替换为box_right_left
                anchor_middle_x = box_right_left
            if anchor_middle_x >= box_left_right and anchor_middle_x <= box_right_left:  # 位于box中段的
                anchor_y_top = calcY(anchor_middle_x, lines[0])
                anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(6)
                continue
            else:
                if lines[3][1] == 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                    y_top.append(anchor_y_top)
                    y_bottom.append(anchor_y_bottom)
                    continue
                k_l34 = -(lines[3][0]/lines[3][1]) 
                if k_l34 > 0:
                    anchor_y_top = calcY(anchor_middle_x, lines[3])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[1])
                else:
                    anchor_y_top = calcY(anchor_middle_x, lines[0])
                    anchor_y_bottom = calcY(anchor_middle_x, lines[3])
                y_top.append(anchor_y_top)
                y_bottom.append(anchor_y_bottom)
                whichline.append(7)
                continue
    
    # print(y_top)
    # print(y_bottom)
    # print(whichline)
    return y_top,y_bottom
        
