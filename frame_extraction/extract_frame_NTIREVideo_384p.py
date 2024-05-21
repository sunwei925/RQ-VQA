import os
import cv2
import pandas as pd
import argparse

def extract_frame(videos_dir, video_name, save_folder):
    filename = os.path.join(videos_dir, video_name)
    print(filename)
    video_name_str = video_name[:-4]
    video_capture = cv2.VideoCapture()
    video_capture.open(filename)
    cap=cv2.VideoCapture(filename)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # the heigh of frames
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # the width of frames

    print(video_height)
    print(video_width)
    
    if video_height > video_width:
        video_width_resize = 384
        video_height_resize = int(video_width_resize/video_width*video_height)
    else:
        video_height_resize = 384
        video_width_resize = int(video_height_resize/video_height*video_width)
        
    dim = (video_width_resize, video_height_resize)

    video_read_index = 0

    frame_idx = 0
    
    video_length_min = 8

    if video_frame_rate !=0:
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length) and (frame_idx % video_frame_rate == 0):
                    read_frame = cv2.resize(frame, dim)
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                            '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                    video_read_index += 1
                frame_idx += 1
                
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(i) + '.png'), read_frame)
    else:
        # to avoid the situation that the frame rate is less than 1 fps
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                # key frame
                if (video_read_index < video_length):
                    read_frame = cv2.resize(frame, dim)
                    exit_folder(os.path.join(save_folder, video_name_str))
                    cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                            '{:03d}'.format(video_read_index) + '.png'), read_frame)          
                    video_read_index += 1
                
        if video_read_index < video_length_min:
            for i in range(video_read_index, video_length_min):
                cv2.imwrite(os.path.join(save_folder, video_name_str, \
                                        '{:03d}'.format(i) + '.png'), read_frame)        


    return
            
def exit_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)    
        
    return



def main(config):
    videos_dir = config.videos_dir
    filename_path = config.filename_path

    column_names = ['filename', 'score']
    dataInfo = pd.read_csv(filename_path, header = 0, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
    video = dataInfo['filename'].tolist()

    video_names = []
    for video_i in video:
        video_names.append(video_i)

    n_video = len(video_names)

    save_folder = config.save_folder
    for i in range(n_video):
        video_name = video_names[i]
        print('start extract {}th video: {}'.format(i, video_name))
        extract_frame(videos_dir, video_name, save_folder)
    
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # input parameters
    parser.add_argument('--videos_dir', type=str)
    parser.add_argument('--filename_path', type=str)
    parser.add_argument('--save_folder', type=str)

    
    config = parser.parse_args()

    main(config)