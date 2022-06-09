# import glob
# import os
# import cv2
import random
import json
import pickle
import numpy as np
from tqdm import tqdm


def get_label_list(pos_score_compute, th, with_pose_list=False):
    label_list = []
    pos_list = []
    other_list = []
    i_p = 0
    # 处理没有相等的情况
    while i_p < len(pos_score_compute):
        if pos_score_compute[i_p] < th:
            i_p += 1
            continue
        # 边界考虑，第0个
        if i_p == 0:
            if pos_score_compute[i_p] > pos_score_compute[i_p + 1]:
                label_list.append(0.25 / 2)
                pos_list.append(0)
        # 考虑最后一个
        elif i_p == (len(pos_score_compute) - 1):
            if pos_score_compute[i_p] > pos_score_compute[i_p - 1]:
                label_list.append(0.25 / 2 + i_p * 0.25)
                pos_list.append(i_p)
        else:
            if (pos_score_compute[i_p] > pos_score_compute[i_p + 1]) and (
                    pos_score_compute[i_p] > pos_score_compute[i_p - 1]):
                label_list.append(0.25 / 2 + i_p * 0.25)
                pos_list.append(i_p)
        i_p += 1
        continue
    # 处理相等情况
    i_p = 0
    while i_p < len(pos_score_compute):
        if pos_score_compute[i_p] < th:
            i_p += 1
            continue
        comput_pos = 0
        count_pos = 0
        while (i_p < (len(pos_score_compute) - 1)) and int(pos_score_compute[i_p] * 10000) == int(
                pos_score_compute[i_p + 1] * 10000):
            comput_pos += i_p
            count_pos += 1
            i_p += 1
        if count_pos > 0:
            comput_pos += i_p
            count_pos += 1
            c_pos = comput_pos / count_pos
            # 这里加 0.25/2,指标f1 = 0.945 -> 0.950
            label_list.append(0.25 / 2 + c_pos * 0.25)
            if int(c_pos) < (c_pos - 0.45):
                other_list.append(int(comput_pos / count_pos) + 1)
            pos_list.append(int(comput_pos / count_pos))
        else:
            i_p += 1
    label_list = sorted(label_list)
    if with_pose_list:
        return sorted(pos_list), other_list
    return label_list


def generate_val_class_label_check_max_f1(label_map, input_dir1, input_dir2, output_dir_pos, output_dir_neg):
    output_json_file_template = '../output/log/val_max_f1_%.2f.json'

    for threadhold in range(-5, 6, 5):
        # for thread in 10,70, 10]:
        output_json = {}
        th = threadhold / 100
        out_f = output_json_file_template % th
        count = 0
        process_count = 0
        print(threadhold)
        for key in label_map:
            process_count += 1
            pos_score_compute = label_map[key]['pos_compute']
           
            label_list = get_label_list(pos_score_compute, th)
            # th = f1_consis_avg
            out_lidst = []
            if len(label_list) < 1:
                # count += 1
                # print(count,key)
                continue

            output_json[key] = sorted(label_list)
            # assert len(pos_list) == len(label_list)
            path_video = label_map[key]['path_video']
            duration = label_map[key]['video_duration']
            # copy_train_class_data2(pos_list, image_template1, image_template2, output_dir_pos, output_dir_neg,duration)

        with open(out_f, 'w') as f:
            json.dump(output_json, f)

    pass


def create_GEBD_class_train_data_raw():
    '''
    :return:
    '''
    positive_map = {}
    negative_map = {}
    not_exist_map = {}

    label_map = {}
    in_pkl_train = '../export/k400_mr345_train_min_change_duration0.3.pkl'
    in_pkl_val = '../export/k400_mr345_val_min_change_duration0.3.pkl'

    with open(in_pkl_train, 'rb') as f:
        dict_raw_train = pickle.load(f, encoding='lartin1')
        label_map.update(dict_raw_train)
    with open(in_pkl_val, 'rb') as f:
        dict_raw_val = pickle.load(f, encoding='lartin1')
        label_map.update(dict_raw_val)
    input_dir1 = 'GEBD_40fps_images_clear_data/train_val_merge4_img/'

    key_list = list(label_map.keys())
    for key in tqdm(key_list):
        single_d = label_map[key]
        timestamps_list = single_d['substages_timestamps']
        f1_consis = single_d['f1_consis']
        f1_consis_avg = single_d['f1_consis_avg']
        path_video = single_d['path_video']
        video_duration = single_d['video_duration']
        # img_count = int(video_duration / 0.25)
        # image_template = os.path.join(input_dir1, path_video.split('/')[-1].replace('.mp4', '') + '=%03dT%03d.jpg')
        pos_score_compute = np.zeros(40, np.float)
        neg_good = []
        if f1_consis_avg < 0.6:
            neg_good = np.where(np.array(f1_consis) < f1_consis_avg)[0].tolist()

        pos_flag_all = np.ones(40, np.int)
        for index, timestamps in enumerate(timestamps_list):
            if index in neg_good:
                continue
            label_f1 = f1_consis[index]
            pos_flag = np.zeros(40, np.int)
            for single_ts in timestamps:
                case = 1  # 1  方案1， 2 方案2
                # 方案1 取中间值
                if case == 1:
                    st = single_ts
                    pos_st = int(st / 0.25)
                    delta = 1
                    if (st - pos_st * 0.25) > 0.1245:
                        delta = 0
                    #
                    use_smooth = True
                    if use_smooth:
                        for get_pos_s in range(pos_st - delta, pos_st + 2 - delta):
                            if get_pos_s >= 0 and get_pos_s < 40:
                                pos_flag[get_pos_s] = 1
                    else:
                        if pos_st >= 0 and pos_st < 40:
                            pos_flag[pos_st] = 1

                # 方案2 不取中间值
                if case == 2:
                    pass
            for ind, v in enumerate(list(pos_flag)):
                if v > 0:
                    pos_score_compute[ind] += label_f1
                else:
                    pos_score_compute[ind] -= label_f1
            pos_flag_all = pos_flag_all * pos_flag
        label_map[key]['pos_compute'] = pos_score_compute
        label_map[key]['pos_compute_merge'] = pos_score_compute
        label_map[key]['single_lable'] = np.array([get_label_list(pos_score_compute, 0)])

    # save data_set
    # save GEBD data file param
    # get_new_data =  False #  True #
    get_new_data = True  #
    if get_new_data:
        val_ratio = 0.05
        data_version = 'V10'
        classes = ['0', '1']  # ['0','1'], ['0','1','other']
        images_dir = 'GEBD_40fps_images'  # in GPU servers data path
        output_data_pkl_root = '../data_split/'
        output_data_pkl_train = output_data_pkl_root + 'train_%s.pkl' % data_version
        output_data_pkl_val = output_data_pkl_root + 'val_%s.pkl' % data_version
        random.shuffle(key_list)
        val_len = int(len(key_list) * val_ratio)
        val_list = key_list[:val_len]
        train_list = key_list[val_len:]

        val_file_map = {'classes': classes, 'img_dir': images_dir}
        train_file_map = {'classes': classes, 'img_dir': images_dir}
        for val_key in val_list:
            val_file_map[val_key] = label_map[val_key]

        for train_key in train_list:
            train_file_map[train_key] = label_map[train_key]
        print('crete train / val file ok!\n    get train %d video\n    get val %d video ' % (len(train_file_map),
                                                                                             len(val_file_map)))
        with open(output_data_pkl_train, 'wb') as f:
            pickle.dump(train_file_map, f)
        with open(output_data_pkl_val, 'wb') as f:
            pickle.dump(val_file_map, f)
        print('===============================================================')


def create_GEBD_pretrain_data():
    label_map = {}
    in_pkl_train = '../export/k400_mr345_train_min_change_duration0.3.pkl'
    in_pkl_val = '../export/k400_mr345_val_min_change_duration0.3.pkl'
    in_test_json = '../eval/test_len.json'
    in_test_v2_txt = '../eval/Kinetics_GEBD_Test_v2.txt'
    with open(in_pkl_train, 'rb') as f:
        dict_raw_train = pickle.load(f, encoding='lartin1')
        label_map.update(dict_raw_train)
    with open(in_pkl_val, 'rb') as f:
        dict_raw_val = pickle.load(f, encoding='lartin1')
        label_map.update(dict_raw_val)

    # with open(in_test_json,'r') as f:
    #     dict_raw_test = json.loads(f.read())
    with open(in_test_v2_txt, 'r') as f:
        lines = f.readlines()
    for line in lines:
        video_name, path_video = line.replace('\n', '').split(',')
        if video_name in label_map:
            print('repeate test in train/val is:' + line)
            continue
        label_map[video_name] = {}
        label_map[video_name]['path_video'] = path_video
        label_map[video_name]['substages_timestamps'] = []
        label_map[video_name]['single_label'] = []
        label_map[video_name]['f1_consis_avg'] = 1.0

    data_version = 'V1'
    images_dir = './GEBD_40fps_images/'  # in GPU servers data path
    output_data_pkl_root = '../data_split/'
    output_data_pkl_pretrain = output_data_pkl_root + 'pretrain_%s.pkl' % data_version

    pretrain_file_map = {'img_dir': images_dir}
    # label_map['img_dir'] = images_dir
    for key in label_map:
        pretrain_file_map[key] = label_map[key]
        pretrain_file_map[key]['single_label'] = []
    print('crete pretrain %d file ok!' % len(pretrain_file_map))
    with open(output_data_pkl_pretrain, 'wb') as f:
        pickle.dump(pretrain_file_map, f)
    print('===============================================================')


def main():
    # 1 pre-training data prepare
    create_GEBD_pretrain_data()
    # 2 train_data_prepare
    create_GEBD_class_train_data_raw()
    pass


if __name__ == '__main__':
    main()
