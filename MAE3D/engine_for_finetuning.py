import math
import sys
import json
from typing import Iterable, Optional

import torch

max_pooling = torch.nn.MaxPool1d(5, stride=1, padding=2)

from timm.data import Mixup
from timm.utils import ModelEma
import numpy as np
import utils

sys.path.append('./tools')
from tools.get_GEBD_class_data import get_label_list


def train_class_batch(model, samples, target, criterion):
    loss_func_mse = torch.nn.MSELoss()
    x_logistic, x_mse = model(samples)
    float_target = target.float()
    loss_logistic = criterion(x_logistic, float_target)
    # print(x_mse.shape)
    # print(float_target.shape)
    # print(x_mse.size())
    # print(float_target.size())
    loss_mse = loss_func_mse(x_mse, float_target)
    loss = loss_mse + loss_logistic
    outputs = (x_mse + torch.sigmoid(x_logistic)) / 2.0

    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, video_names) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            out = output.unsqueeze(1)
            peak = (out == max_pooling(out))
            peak[out < args.threshold] = False
            peak = peak.int()
            peak = peak.squeeze(1)
            # output_m = output_m[-1]
            class_acc = (peak == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, val_data_pkl):
    criterion = torch.nn.BCEWithLogitsLoss()
    loss_func_mse = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    eval_pose_map = {}
    eval_pose_map_max_pool = {}
    for i_t in range(5, 70, 5):
        key = i_t
        eval_pose_map_max_pool[key] = {}

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        video_names = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            x_logistic, x_mse = model(images)
            float_target = target.float()
            loss_logistic = criterion(x_logistic, float_target)
            loss_mse = loss_func_mse(x_mse, float_target)
            loss = loss_mse + loss_logistic
            output = (x_mse + torch.sigmoid(x_logistic)) / 2.0

        if args.data_reverse:
            out = torch.flip(output, dims=[1])
            output = out

        out = output.unsqueeze(1)

        peak = (out == max_pooling(out))
        peak[out < args.threshold] = False
        peak = peak.squeeze(1)
        peak = peak.int()
        acc = (peak == target).float().mean()

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        # get F1 prepare 1
        np_array = output.cpu().numpy()
        for id, video_n in enumerate(video_names):
            if 'reverse' in video_n:
                video_n_old = video_n.split('=')[0]
                revers_array = np_array[id]
                new_array = revers_array[::-1]
                if video_n_old in eval_pose_map:
                    res_array = revers_array + new_array
                    eval_pose_map[video_n_old] = res_array / 2.0
                else:
                    eval_pose_map[video_n_old] = new_array
                continue
            if video_n in eval_pose_map:
                eval_pose_map[video_n] = (np_array[id] + eval_pose_map[video_n]) / 2.0
            else:
                eval_pose_map[video_n] = np_array[id]
        out = output.unsqueeze(1)
        for thres_d in eval_pose_map_max_pool:
            score = thres_d / 100
            peak_bk = (out == max_pooling(out))
            peak_bk[out < score] = False
            # for id, video_n in enumerate(video_names):

            peak_bk = peak_bk.squeeze(1)
            peak_bk_pos = torch.nonzero(peak_bk).cpu().numpy()

            for i, v in peak_bk_pos:
                video_n = video_names[i]
                # maxpool 暂时不评估reverse
                if 'reverse' in video_n:
                    continue
                if video_n not in eval_pose_map_max_pool[thres_d]:
                    eval_pose_map_max_pool[thres_d][video_n] = []
                eval_pose_map_max_pool[thres_d][video_n].append(v * 0.25 + 0.125)
                # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # get F1 prepare 2
    out_f1_map = {}
    out_f1_map_max_pool = {}
    out_f1_max = 0.0
    out_f1_max_max_pool = 0.0
    for i_f in eval_pose_map_max_pool:
        th = i_f / 100
        th_map = {}
        for key in eval_pose_map:
            th_map[key] = get_label_list(eval_pose_map[key], th)
        out_f1_map[th] = compute_f1_scores(val_data_pkl, th_map, th)
        out_f1_map_max_pool[th] = compute_f1_scores(val_data_pkl, eval_pose_map_max_pool[i_f], th)
        out_f1_max = max(out_f1_max, out_f1_map[th]['f1'])
        out_f1_max_max_pool = max(out_f1_max_max_pool, out_f1_map_max_pool[th]['f1'])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc, losses=metric_logger.loss))
    print('=== total: %d videos ,f1 max:%.3f ,f1 map ==================\n %s' % (len(eval_pose_map), out_f1_max,
                                                                                 json.dumps(out_f1_map)))
    print('=== total: %d videos ,f1 max:%.3f ,f1 max_pool map ==================\n %s' % (len(eval_pose_map),
                                                                                          out_f1_max_max_pool,
                                                                                          json.dumps(
                                                                                              out_f1_map_max_pool)))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_f1_scores(gt_dict, pre_dict, t):
    d = 0.05
    tp_all = 0
    num_pos_all = 0
    num_det_all = 0
    gt_set = set(list(gt_dict.keys()))
    pre_set = set(list(pre_dict.keys()))
    share = gt_set & pre_set
    print('share len is: %d, thread is: %.3f' % (len(share), t))
    bad_case_list = []

    for vid_id in list(gt_dict.keys()):

        # filter by avg_f1 score
        if gt_dict[vid_id]['f1_consis_avg'] < 0.3:
            continue

        if vid_id not in pre_dict:
            continue

        bdy_idx_list_smt = pre_dict[vid_id]

        myfps = gt_dict[vid_id]['fps']
        ins_start = 0
        ins_end = gt_dict[vid_id]['video_duration']

        # remove detected boundary outside the action instance
        tmp = []
        for det in bdy_idx_list_smt:
            tmpdet = det + ins_start
            if tmpdet >= (ins_start) and tmpdet <= (ins_end):
                tmp.append(tmpdet)
        bdy_idx_list_smt = tmp
        if bdy_idx_list_smt == []:
            num_pos_all += len(gt_dict[vid_id]['substages_timestamps'][0])
            continue
        num_det = len(bdy_idx_list_smt)
        num_det_all += num_det

        # compare bdy_idx_list_smt vs. each rater's annotation, pick the one leading the best f1 score
        bdy_idx_list_gt_allraters = gt_dict[vid_id]['substages_timestamps']
        f1_tmplist = np.zeros(len(bdy_idx_list_gt_allraters))
        tp_tmplist = np.zeros(len(bdy_idx_list_gt_allraters))
        num_pos_tmplist = np.zeros(len(bdy_idx_list_gt_allraters))

        for ann_idx in range(len(bdy_idx_list_gt_allraters)):
            bdy_idx_list_gt = bdy_idx_list_gt_allraters[ann_idx]
            num_pos = len(bdy_idx_list_gt)
            tp = 0
            offset_arr = np.zeros((len(bdy_idx_list_gt), len(bdy_idx_list_smt)))
            for ann1_idx in range(len(bdy_idx_list_gt)):
                for ann2_idx in range(len(bdy_idx_list_smt)):
                    offset_arr[ann1_idx, ann2_idx] = abs(bdy_idx_list_gt[ann1_idx] - bdy_idx_list_smt[ann2_idx])
            for ann1_idx in range(len(bdy_idx_list_gt)):
                if offset_arr.shape[1] == 0:
                    break
                min_idx = np.argmin(offset_arr[ann1_idx, :])
                if offset_arr[ann1_idx, min_idx] <= d * (ins_end - ins_start):
                    tp += 1
                    offset_arr = np.delete(offset_arr, min_idx, 1)

            num_pos_tmplist[ann_idx] = num_pos
            fn = num_pos - tp
            fp = num_det - tp
            if num_pos == 0:
                rec = 1
            else:
                rec = tp / (tp + fn)
            if (tp + fp) == 0:
                prec = 0
            else:
                prec = tp / (tp + fp)
            if (rec + prec) == 0:
                f1 = 0
            else:
                f1 = 2 * rec * prec / (rec + prec)
            tp_tmplist[ann_idx] = tp
            f1_tmplist[ann_idx] = f1

        ann_best = np.argmax(f1_tmplist)
        f1 = np.max(f1_tmplist)
        if f1 < 0.3:
            bad_case_list.append(vid_id)
            # print(len(bad_case_list),vid_id)
        tp_all += tp_tmplist[ann_best]
        num_pos_all += num_pos_tmplist[ann_best]

    fn_all = num_pos_all - tp_all
    fp_all = num_det_all - tp_all
    if num_pos_all == 0:
        rec = 1
    else:
        rec = tp_all / (tp_all + fn_all)
    if (tp_all + fp_all) == 0:
        prec = 0
    else:
        prec = tp_all / (tp_all + fp_all)
    if (rec + prec) == 0:
        f1 = 0
    else:
        f1 = 2 * rec * prec / (rec + prec)
    return {'recall': rec, 'precision': prec, 'f1': f1}
