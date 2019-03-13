import numpy as np 

def cpu_nms(dets, thresh): 
    """Pure Python NMS baseline.""" 
    x1 = dets[:, 0] 
    y1 = dets[:, 1] 
    x2 = dets[:, 2] 
    y2 = dets[:, 3] 
    scores = dets[:, 4] 

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    #按照从小到大排序后返回下标，然后顺序取反，即从大到小对应的下标 
    order = scores.argsort()[::-1]

    keep = [] 
    while order.size > 0: 
        i = order[0] 
        keep.append(i) 
        #求交叉面积intersection采用了这个非常巧妙的方法，自己画一下思考一下 
        xx1 = np.maximum(x1[i], x1[order[1:]]) 
        yy1 = np.maximum(y1[i], y1[order[1:]]) 
        xx2 = np.minimum(x2[i], x2[order[1:]]) 
        yy2 = np.minimum(y2[i], y2[order[1:]]) 
        
        w = np.maximum(0.0, xx2 - xx1 + 1) #计算w 
        h = np.maximum(0.0, yy2 - yy1 + 1) #计算h 
        inter = w * h #交叉面积 
        #A交B/A并B 
        ovr = inter / (areas[i] + areas[order[1:]] - inter) 
        """
        保留重叠面积小于threshold的
        np.where的返回值是tuple
        第一个维度是x的list，第二个维度是y的list
        这里因为输入是1维，因此就取0就好
        """ 
        inds = np.where(ovr <= thresh)[0] 
        order = order[inds + 1] 
    return keep