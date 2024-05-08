import os.path
import cv2
import pymysql
import numpy as np
from sklearn.metrics import recall_score
from collections import Counter
from statistics import mean
import surface_distance as surfdist

def cal_confu_matrix(label, predict, class_num):
    #获得混淆矩阵
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)


def find_contours(targets, pred):
    #过检面积的阈值需要确定
    targets_num = 0
    pred_num = 0

    targets = cv2.bitwise_not(targets)
    pred = cv2.bitwise_not(pred)
    contours1, _ = cv2.findContours(targets, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours1_new = []
    contours2_new = []
    for contour1 in contours1:
        if cv2.contourArea(contour1) > 50:
            contours1_new.append(contour1)
    for contour2 in contours2:
        if cv2.contourArea(contour2) > 10:
            contours2_new.append(contour2)
    if len(contours1_new) == 0 and len(contours2_new) == 0:
        # 如果预测和label全是空的情况
        targets_num = 0
        pred_right_num = 0
        pred_right_num_ratio = 1
        pred_over_num = 0
    elif len(contours1_new) != 0 and len(contours2_new) == 0:
        # 如果label不为0，但是预测为0的情况，即漏检
        targets_num = len(contours1_new)
        pred_right_num = 0
        pred_right_num_ratio = 0
        pred_over_num = 0
    elif len(contours1_new) == 0 and len(contours2_new) != 0:
        # 如果label为0，但是预测不为0的情况，即过检
        targets_num = 0
        pred_right_num = 0
        pred_right_num_ratio = 0
        pred_over_num = len(contours2_new)
    else:
        for contour1 in contours1_new:
            if cv2.contourArea(contour1) > 50:
                targets_num += 1
        pred_right_num = 0
        for contour2 in contours2_new:
            if cv2.contourArea(contour2) > 10:
                pred_num += 1

            mask = np.zeros_like(pred)
            cv2.drawContours(mask, [contour2], -1, (255, 255, 255), thickness=-1)
            extracted_regions = cv2.bitwise_and(pred, mask)
            result = cv2.bitwise_and(targets, extracted_regions)

            mask0 = result > 0

            # 获取非零像素的位置
            non_zero_pixels = np.transpose(np.nonzero(mask0))
            if len(non_zero_pixels) != 0:
                pred_right_num += 1
                continue
        pred_over_num = pred_num - pred_right_num
        pred_right_num_ratio = pred_right_num / targets_num
    return targets_num, pred_right_num, pred_right_num_ratio, pred_over_num


def seg_parameter(label_path, pre_path, num_class):
    targets_num_list = []
    pred_right_num_list = []
    pred_right_num_ratio_list = []
    pred_over_num_list = []
    PA_list = []
    reacll_list = []
    IOU_list = []
    f1_score_list = []

    ASSD_list = []
    Dice_list = []


    for file in os.listdir(label_path):
        print(file)
        if file.split(".")[-1] =="db":
            continue
        label_name = os.path.join(label_path, file)
        pre_name = os.path.join(pre_path, file)
        # pre_name = os.path.join(pre_path, file.split("_label")[0]+".png")
        targets = cv2.imdecode(np.fromfile(label_name, dtype=np.uint8), 0)
        targets = (targets > 125).astype(np.uint8) * 255

        preds = cv2.imdecode(np.fromfile(pre_name, dtype=np.uint8),0)
        preds = (preds > 125).astype(np.uint8) * 255

        # 计算ASSD
        gt = targets.astype(np.bool)
        pred = preds.astype(np.bool)
        vertical = 1
        horizontal = 1
        surface_distances = surfdist.compute_surface_distances(gt, pred, spacing_mm=(vertical, horizontal))
        # avg_surf_dist有两个参数，第一个参数是average_distance_gt_to_pred，第二个参数是average_distance_pred_to_gt
        surf_dist = surfdist.compute_average_surface_distance(surface_distances)
        avg_surf_dist = (surf_dist[0] + surf_dist[1]) / 2
        if np.isnan(avg_surf_dist):
            continue
        ASSD_list.append(avg_surf_dist)
        surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 5)
        Dice_list.append(surface_dice)
        # print('assd:',ASSD_list)


        targets_num, pred_right_num, pred_right_num_ratio, pred_over_num = find_contours(targets, preds)
        targets_num_list.append(targets_num)
        pred_right_num_list.append(pred_right_num)
        pred_right_num_ratio_list.append(pred_right_num_ratio)
        pred_over_num_list.append(pred_over_num)
        if pred_right_num == 0 and pred_right_num_ratio==1:
            # 如果预测和label全是空的情况
            PA = 1
            reacll = 1
            IOU = 1
            f1_score = 1
        elif pred_right_num == 0 and targets_num_list != 0:
            # 如果label不为0，但是预测为0的情况，即漏检
            PA = 0
            reacll = 0
            IOU = 0
            f1_score = 0
        elif pred_right_num != 0 and targets_num_list == 0:
            # 如果label为0，但是预测不为0的情况，即过检
            PA = 0
            reacll = 0
            IOU = 0
            f1_score = 0
        else:
            targets = targets / 255
            preds = preds / 255

            confu_mat_total = cal_confu_matrix(targets, preds, num_class)
            class_num = confu_mat_total.shape[0]
            confu_mat = confu_mat_total.astype(np.float32)
            col_sum = np.sum(confu_mat, axis=1)  # 按行求和
            raw_sum = np.sum(confu_mat, axis=0)  # 每一列的数量

            pe_fz = 0
            PA = 0  # 像素准确率
            CPA = []  # 类别像素准确率
            TP = []  # 识别中每类分类正确的个数

            for i in range(class_num):
                pe_fz += col_sum[i] * raw_sum[i]
                PA = PA + confu_mat[i, i]
                CPA.append(confu_mat[i, i] / col_sum[i])
                TP.append(confu_mat[i, i])
            #准确率precision_
            PA = PA / confu_mat.sum()
            CPA = np.array(PA)
            PA = np.mean(CPA)  # 类别平均像素准确率

            # 计算f1-score
            TP = np.array(TP)
            FN = col_sum - TP
            FP = raw_sum - TP

            # 计算并写出f1_score,IOU,Mf1,MIOU
            f1_score = []  # 每个类别的f1_score

            IOU = []  # 每个类别的IOU
            for i in range(class_num):
                # 写出f1-score
                f1 = TP[i] * 2 / (TP[i] * 2 + FP[i] + FN[i])
                f1_score.append(f1)
                iou = TP[i] / (TP[i] + FP[i] + FN[i])
                IOU.append(iou)
            # iou
            IOU = np.array(IOU)
            IOU = np.mean(IOU)

            #f1_score
            f1_score = np.array(f1_score)
            f1_score = np.mean(f1_score)

            #reacll召回率
            reacll = recall_score(preds, targets, average="micro")
            # 多分类的召回率， labels类别的值，pos_label正样本的值
            # reacll = recall_score(pre, label, labels = [0, 1], pos_label= 1, average="micro")

        PA_list.append(PA)
        reacll_list.append(reacll)
        IOU_list.append(IOU)
        f1_score_list.append(f1_score)

    return mean(Dice_list),mean(ASSD_list),mean(PA_list), mean(reacll_list), mean(IOU_list), mean(f1_score_list), sum(targets_num_list), sum(pred_right_num_list), mean(pred_right_num_ratio_list), sum(pred_over_num_list)


if __name__ == '__main__':
    # resnet
    # model_name = 'resnet'
    # label_path = './val_preds/fcn_resnet50_params/labels'
    # pre_path = './val_preds/fcn_resnet50_params/preds'
    # # unetpp
    # model_name = 'unetpp'
    # label_path = './val_preds/NestedUNet_params/labels'
    # pre_path = './val_preds/NestedUNet_params/preds'
    # # unet
    model_name = 'SGUNet'
    label_path = './val_preds/labels'
    pre_path = './val_preds/preds'
    num_class = 2
    DICE,ASSD, PA, reacll, IOU, f1_score, targets_num, pred_right_num, pred_right_num_ratio, pred_over_num = seg_parameter(
            label_path, pre_path, num_class)
    with open("evaluate_res.txt", "a") as f:
        f.writelines(model_name)
        f.writelines("\n")
        f.writelines('DICE: '+str(DICE)+' '+'ASSD: '+str(ASSD)+' '+'PA: '+str(PA)+' '+'reacll: '+str(reacll)+' '+'f1_score: '+str(f1_score)+' ')
        f.writelines("\n")

    print(ASSD, PA,  reacll, f1_score)