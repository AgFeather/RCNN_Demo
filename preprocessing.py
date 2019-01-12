import skimage
import selectivesearch
import numpy as np
from PIL import Image
import pickle

import utils


# IOU 概念：IOU定义了两个bounding box的重叠度，计算方式为两个box的交集除以两个box的并集
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    # 通过四条if来查看两个方框是否有交集。如果四种状况都不存在，我们视为无交集
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False

    # 如果相交，计算相交的面积并返回
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmin_b, xmax_a, xmax_b])
        y_sorted_list = sorted([ymin_a, ymin_b, ymax_a, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_intersect = x_intersect_w * y_intersect_h
        return area_intersect

def IOU(ver1, vertice2):
    # 整理四个顶点
    vertice1 = [ver1[0], ver1[1], ver1[0] + ver1[2], ver1[1] + ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3],
                                 vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    # 如果有交集，计算IOU
    if area_inter:
        area1 = ver1[2] * ver1[3]
        area2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area1 + area2 - area_inter)
        return iou
    return False

def clip_pic(img, rect):
    # 将原始图像根据候选框的大小进行切割，并返回切割后的小图像
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]

def load_train_proposals(data_file, num_class, threshold=0.5, svm=True, save=True, save_path='split_dataset.pkl'):
    # 在fine tune Alexnet时以0.5位IOU的threthold，在训练SVM时以0.3为threthold
    train_list = open(data_file, 'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        # tmp0 = image address
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = skimage.io.imread(tmp[0])
        # 使用selective search函数生成候选框
        img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
        candidates = set()
        for r in regions:
            if r['rect'] in candidates:  # 剔除重复的候选框
                continue
            if r['size'] < 220:  # 提出小的候选框
                continue
            x, y, w, h = r['rect']
            if w == 0 or h == 0:  # 长或者宽为0，则剔除
                continue

            # 按照当前候选框的大小对原始图片进行切割
            proposal_img, proposal_vertice = clip_pic(img, r['rect'])
            if len(proposal_img) == 0:  # 如果截取后的图片为None，删除
                continue
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:  # image array的dim里有0的，剔除
                continue
            im = Image.fromarray(proposal_img)
            resized_proposal_img = utils.resize_image(im, 224, 224)
            candidates.add(r['rect'])

            # 计算IOU
            ref_rect = tmp[2].split(',')
            ref_rect_int = [int(i) for i in ref_rect]
            iou_val = IOU(ref_rect_int, proposal_vertice)
            index = int(tmp[1])
            if svm == False:  # svm为False时使用one-hot encoding
                label = np.zeros(num_class + 1)
                if iou_val < threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
            else:
                if iou_val < threshold:
                    labels.append(0)
                else:
                    labels.append(index)
    if save:
        pickle.dump((images, labels), open(save_path, 'wb'))
        print(save_path, 'has saved...')
    return images, labels


if __name__ == '__main__':
    X, Y = load_train_proposals('refine_list.txt', 2, save=True)
