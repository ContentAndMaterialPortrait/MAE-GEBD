import os
import glob
from tqdm import tqdm


def get_video2img_for_mae():
    '''
    use: conda activate GEBD && python prepare_data_MAE_2fps_img.py
    '''
    # train
    in_dir_list = ['kinetics-datasets/GEBD_all_mp4/']
    output_dir = 'cvpr2022-workshop/GEBD_40fps_images'

    input_mp4_list = []
    for mp4_dir in in_dir_list:
        mp4_list = glob.glob(mp4_dir + '*.mp4')
        input_mp4_list.extend(mp4_list)
    print('has get total mp4 files: %d ' % len(input_mp4_list))
    sort_input_mp4 = sorted(input_mp4_list)
    total_cnt = len(sort_input_mp4)
    total_part = 1
    step = total_cnt // total_part
    part = 0
    start = part * step
    if part == total_part - 1:
        end = total_cnt
    else:
        end = (part + 1) * step

    print('===================part:%d / %d, ===st/end: %d / %d =====================' % (part, total_part, start, end))
    command = 'ffmpeg -loglevel quiet -i %s -vsync 2 -f image2 -r 4 %s'
    count = 0
    cnt_nu = 38
    for mp4_file in tqdm(sort_input_mp4[start:end]):
        img_name = mp4_file.split('/')[-1].replace('.mp4', '') + '=%03d.jpg'
        output_path_cnt = os.path.join(output_dir, img_name) % cnt_nu
        if not os.path.exists(output_path_cnt):
            # print(output_path_cnt)
            output_path = os.path.join(output_dir, img_name)
            os.system(command % (mp4_file, output_path))
            count += 1

    print('has loss data mp4 count is:%d' % count)

    pass


if __name__ == '__main__':
    get_video2img_for_mae()

    pass
