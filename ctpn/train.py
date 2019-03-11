import torch.optim as optim  # 优化器
import torch  # PyTorch包
import cv2  # OpenCV
import lib.tag_anchor # 用来标记是正样本还是负样本
import lib.generate_gt_anchor  # 生成实际（ground truth）锚框，作为训练、测试样本
import lib.dataset_handler  # 图像预处理
import lib.utils  # 小工具类，画框、图像base64互转、初始化权重等
import numpy as np  #numpy库
import os  # 输入输出库
import Net.net as Net  # CTPN网络类
import Net.loss as Loss  # 损失函数类
import configparser  # 命令行参数解析库，python自带
import time  # 时间库，python自带
import evaluate
import logging  # 输出日志库，python自带
import datetime  # 时间库，python自带
import copy  
import random  # 随机库，打乱输入数据顺序之用（shuffle）
import matplotlib.pyplot as plt  # matplotlib绘图库

DRAW_PREFIX = './anchor_draw'
# MSRA = '/home/sambauser/ctpn_batch_num/data_ready/MSRA_TD500'  # MSRA_TD500数据集路径
ALI = '/home/sambauser/ctpn_batch_num/data/ali_icpr' # ali_icpr数据集路径
# DATASET_LIST = [MSRA, ALI]
DATASET_LIST = [ALI]
MODEL_SAVE_PATH = './model'


# 遍历文件夹，将文件名组成列表返回
def loop_files(path):
    files = []
    l = os.listdir(path)
    for f in l:
        files.append(os.path.join(path, f))
    return files


# 构建训练、测试数据
def create_train_val():
    train_im_list = []
    test_im_list = []
    train_gt_list = []
    test_gt_list = []
    for dataset in DATASET_LIST:
        trains_im_path =os.path.join(dataset, 'train_im')
        tests_im_path = os.path.join(dataset, 'test_im')
        trains_gt_path =os.path.join(dataset, 'train_gt')
        test_gt_path = os.path.join(dataset, 'test_gt')
        train_im = loop_files(trains_im_path)
        train_gt = loop_files(trains_gt_path)
        test_im = loop_files(tests_im_path)
        test_gt = loop_files(test_gt_path)
        train_im_list += train_im
        test_im_list += test_im
        train_gt_list += train_gt
        test_gt_list += test_gt
    return train_im_list, train_gt_list, test_im_list, test_gt_list


# 绘制loss变化图，包含了train loss和test loss
def draw_loss_plot(train_loss_list=[], test_loss_list=[]):
    x1 = range(0, len(train_loss_list))
    x2 = range(0, len(test_loss_list))
    y1 = train_loss_list
    y2 = test_loss_list
    plt.switch_backend('agg')
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('train loss vs. iterators')
    plt.ylabel('train loss')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('test loss vs. iterators')
    plt.ylabel('test loss')
    plt.savefig("test_train_loss.jpg")


if __name__ == '__main__':
    #########################################
    ################ 加载超参 ################
    #########################################
    cf = configparser.ConfigParser()
    cf.read('./config')  # 读取超参，是否使用GPU、迭代次数、模型保存频率，batch、学习率、epoch

    log_dir = './logs_10'  # 日志保存文件？不存在的话自动创建

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    log_file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log'
    log_handler = logging.FileHandler(os.path.join(log_dir, log_file_name), 'w')
    log_format = formatter = logging.Formatter('%(asctime)s: %(message)s')
    log_handler.setFormatter(log_format)
    logger.addHandler(log_handler)

    gpu_id = cf.get('global', 'gpu_id')
    epoch = cf.getint('global', 'epoch')  # 整个训练集输入训练为一个epoch，总共几个epoch
    val_batch_size = cf.getint('global', 'val_batch')
    logger.info('Total epoch: {0}'.format(epoch))

    using_cuda = cf.getboolean('global', 'using_cuda')
    display_img_name = cf.getboolean('global', 'display_file_name')
    display_iter = cf.getint('global', 'display_iter')  # 输出训练情况的迭代次数设置
    val_iter = cf.getint('global', 'val_iter')  # 验证模型的迭代次数设置
    save_iter = cf.getint('global', 'save_iter')  # 保存模型的迭代次数设置

    lr_front = cf.getfloat('parameter', 'lr_front')
    lr_behind = cf.getfloat('parameter', 'lr_behind')
    change_epoch = cf.getint('parameter', 'change_epoch') - 1  # 修改学习率的epoch设置，前change_epoch-1为front学习率，change_epoch及以后为behind学习率
    logger.info('Learning rate: {0}, {1}, change epoch: {2}'.format(lr_front, lr_behind, change_epoch + 1))
    print('Using gpu id(available if use cuda): {0}'.format(gpu_id))
    print('Train epoch: {0}'.format(epoch))
    print('Use CUDA: {0}'.format(using_cuda))

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    no_grad = [
        'cnn.VGG_16.convolution1_1.weight',
        'cnn.VGG_16.convolution1_1.bias',
        'cnn.VGG_16.convolution1_2.weight',
        'cnn.VGG_16.convolution1_2.bias'
    ]

    if not os.path.exists(MODEL_SAVE_PATH):
        os.mkdir(MODEL_SAVE_PATH)

    ################################################################
    ################ 构造网络结构和Loss，训练过程求出 ################
    ################################################################

    # 这里构造网络，CTPN = VGG16 + RNN + FC
    # 训练:优化器我们选择SGD，learning rate我们设置了两个，前N个epoch使用较大的lr，后面的epoch使用较小的lr以更好地收敛。
    # 训练过程我们定义了4个loss，分别是total_cls_loss，total_v_reg_loss， total_o_reg_loss， total_loss（前面三个loss相加）。
    net = Net.CTPN()
    for name, value in net.named_parameters():  # 这判断哪些参数参与训练，另外一些不参与训练？
        if name in no_grad:
            value.requires_grad = False
        else:
            value.requires_grad = True
    # for name, value in net.named_parameters():
    #     print('name: {0}, grad: {1}'.format(name, value.requires_grad))
    net.load_state_dict(torch.load('./lib/vgg16.model'))  # 加载预训练好的模型
    # net.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    lib.utils.init_weight(net)
    if using_cuda:
        net.cuda()
    net.train()
    print(net)

    criterion = Loss.CTPN_Loss(using_cuda=using_cuda)  # 获取loss

    train_im_list, train_gt_list, val_im_list, val_gt_list = create_train_val()  # 获取所有训练、测试数据文件地址列表
    total_iter = len(train_im_list)
    print("total training image num is %s" % len(train_im_list))
    print("total val image num is %s" % len(val_im_list))

    train_loss_list = []
    test_loss_list = []

    #########################################
    ################ 网络训练 ################
    #########################################
    for i in range(epoch):
        if i >= change_epoch:
            lr = lr_behind
        else:
            lr = lr_front
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)  # SGD优化器
        #optimizer = optim.Adam(net.parameters(), lr=lr)
        iteration = 1  # 现在的迭代次数
        total_loss = 0  # 总的loss
        total_cls_loss = 0  # 分类loss
        total_v_reg_loss = 0  # yh回归loss
        total_o_reg_loss = 0  # side回归loss
        start_time = time.time()

        random.shuffle(train_im_list)  # 打乱训练集
        # print(random_im_list)
        for im in train_im_list: # 每次只训练一张照片么？？？
            root, file_name = os.path.split(im)  # 拆成路径和带后缀图片名
            root, _ = os.path.split(root)  # root为图片所在文件夹
            name, _ = os.path.splitext(file_name)  # name为不带后缀文件名
            # gt_name = 'gt_' + name + '.txt'  # 构造gt_name.txt
            gt_name = name + '.txt'  # 构造gt_name.txt

            gt_path = os.path.join(root, "train_gt", gt_name)  # root_train_gt_gt_name

            if not os.path.exists(gt_path):  # 如果图片对应的label不存在的话跳过
                print('Ground truth file of image {0} not exists.'.format(im))
                continue

            gt_txt = lib.dataset_handler.read_gt_file(gt_path)  # 读取对应的标签，把标签转为list返回，每个元素为一个bbox的4个坐标
            #print("processing image %s" % os.path.join(img_root1, im))
            img = cv2.imread(im) # 把图片读取为opencv对象
            if img is None:  # 图片不存在的话返回
                iteration += 1
                continue

            img, gt_txt = lib.dataset_handler.scale_img(img, gt_txt)  # 图像缩放，保证最短边为600，标签也同步缩放
            tensor_img = img[np.newaxis, :, :, :]  # 将图片数据变成tensor的一条记录，加一个中括号
            tensor_img = tensor_img.transpose((0, 3, 1, 2))  # 按照torch对图片的格式要求，修改图片的轴，从[h,w,c]——>[c,h,w]
            if using_cuda:
                tensor_img = torch.FloatTensor(tensor_img).cuda()
            else:
                tensor_img = torch.FloatTensor(tensor_img)  # 将img像素值转为float的tensor
            
            # tensor_img       [1, 3, 600, 919]
            # vertical_pred    [1, 20, 37, 57]
            # score            [1, 20, 37, 57]
            # side_refinement  [1, 10, 37, 57]
            # 原图片中锚框大小为16*16，网络对每个锚框的10种子锚框进行预测，包含了其坐标yh和前景、背景得分，边框回归值
            vertical_pred, score, side_refinement = net(tensor_img)  # 正向计算，获取预测结果
            del tensor_img

            # 把bbox ground truth转成用于训练的anchor ground truth
            # transform bbox gt to anchor gt for training
            positive = []
            negative = []
            vertical_reg = []
            side_refinement_reg = []

            visual_img = copy.deepcopy(img)  # 该图用于可视化标签

            try:
                # loop all bbox in one image
                # 遍历一张图片中的所有bbox
                # 这里很花时间，平均需要花费6s一张图
                for box in gt_txt:  # gt_txt为实际样本图片中的所有bbox的四个坐标
                    # 从一个bbox中生成anchors，生成一小条一小条竖的anchor
                    # generate anchors from one bbox
                    # 获取图像的anchor标签
                    gt_anchor, visual_img = lib.generate_gt_anchor.generate_gt_anchor(img, box, draw_img_gt=visual_img)  
                    # 计算预测值反映在anchor层面的数据，可以理解为将预测值转为anchor的属性
                    # 有了真实的一小条anchor，加上网络对各个锚框10种尺寸anchor的score，就能从默认的10种尺寸锚框中区分出正样本和负样本，
                    # 垂直回归y和h的缩放比率，边框缩放比率
                    positive1, negative1, vertical_reg1, side_refinement_reg1 = lib.tag_anchor.tag_anchor(gt_anchor, score, box)  
                    positive += positive1
                    negative += negative1
                    vertical_reg += vertical_reg1
                    side_refinement_reg += side_refinement_reg1
            except:
                print("warning: img %s raise error!" % im)
                iteration += 1
                continue

            if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
                iteration += 1
                continue

            cv2.imwrite(os.path.join(DRAW_PREFIX, file_name), visual_img)
            optimizer.zero_grad()
            # 计算成本函数Loss，score、vertical_pre、side_refinement为网络预测值，positive、negative、vertical_reg、side_refinement_reg为实际值
            loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                               negative, vertical_reg, side_refinement_reg)
            # 通过Loss反向传播
            loss.backward()
            optimizer.step()
            iteration += 1
            # save gpu memory by transferring loss to float
            total_loss += float(loss)
            total_cls_loss += float(cls_loss)
            total_v_reg_loss += float(v_reg_loss)
            total_o_reg_loss += float(o_reg_loss)

            if iteration % display_iter == 0:
                end_time = time.time()
                total_time = end_time - start_time
                print('Epoch: {2}/{3}, Iteration: {0}/{1}, loss: {4}, cls_loss: {5}, v_reg_loss: {6}, o_reg_loss: {7}, {8}'.
                      format(iteration, total_iter, i, epoch, total_loss / display_iter, total_cls_loss / display_iter,
                             total_v_reg_loss / display_iter, total_o_reg_loss / display_iter, im))

                logger.info('Epoch: {2}/{3}, Iteration: {0}/{1}'.format(iteration, total_iter, i, epoch))
                logger.info('loss: {0}'.format(total_loss / display_iter))
                logger.info('classification loss: {0}'.format(total_cls_loss / display_iter))
                logger.info('vertical regression loss: {0}'.format(total_v_reg_loss / display_iter))
                logger.info('side-refinement regression loss: {0}'.format(total_o_reg_loss / display_iter))

                train_loss_list.append(total_loss)

                total_loss = 0
                total_cls_loss = 0
                total_v_reg_loss = 0
                total_o_reg_loss = 0
                start_time = time.time()

            # 定期验证模型性能
            if iteration % val_iter == 0:
                net.eval()
                logger.info('Start evaluate at {0} epoch {1} iteration.'.format(i, iteration))
                val_loss = evaluate.val(net, criterion, val_batch_size, using_cuda, logger, val_im_list)  # 验证集评估
                logger.info('End evaluate.')
                net.train()
                start_time = time.time()
                test_loss_list.append(val_loss)

            # 定期存储模型
            if iteration % save_iter == 0:
                print('Model saved at ./model/ctpn-{0}-{1}.model'.format(i, iteration))
                torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-msra_ali-{0}-{1}.model'.format(i, iteration)))

        print('Model saved at ./model/ctpn-{0}-end.model'.format(i))
        torch.save(net.state_dict(), os.path.join(MODEL_SAVE_PATH, 'ctpn-msra_ali-{0}-end.model'.format(i)))

        # 画出loss的变化图
        draw_loss_plot(train_loss_list, test_loss_list)
