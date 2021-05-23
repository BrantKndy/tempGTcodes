import cv2 as cv
import os
import pycocotools
import numpy as np

label_path = ''
save_path = ''
val = ['0002', '0006']
COLORS = ()

for seq in val:
    with open(os.path.join(label_path, seq+'txt'), 'r')as f:
        last_frame = -1
        rle_list = []
        trackid_list = []
        line = f.readline()
        while(line):
            frame, trackid, classid, h, w, rle = line.split()
            if frame == last_frame:
                rle_list.append(rle)
                trackid_list.append(trackid)
            else:
                vis_label = np.zeros((640, 2448, 3))
                for trackid_vis, rle_vis in zip(trackid_list, rle_list):
                    mask = pycocotools.encoder(rle_vis)
                    vis_label[mask != 0] = COLORS[int(trackid_vis)]
                cv.imwrite(os.path.join(save_path, seq+'_'+frame+'.png'), vis_label)
                rle_list = [rle]
                trackid_list = [trackid]
                last_frame = frame
        vis_label = np.zeros((640, 2448, 3))
        for trackid_vis, rle_vis in zip(trackid_list, rle_list):
            mask = pycocotools.encoder(rle_vis)
            vis_label[mask != 0] = COLORS[int(trackid_vis)]
        cv.imwrite(os.path.join(save_path, seq+'_'+frame+'.png'), vis_label)
