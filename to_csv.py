import json
import os
import numpy as np
import csv
import cv2
from tqdm import tqdm
datasets = ['./adas/route1', './adas/route2', './adas/route3', './adas/route4', './adas/route5', './adas/route6', './adas/route7', './adas/route8']
seg_anns = []
classes = [
    "parking_side",
    "parking_marked"
]
for dataset in datasets:
    img_list = list(os.listdir(os.path.join(dataset, 'img')))
    for img_name in tqdm(img_list):
        ann_name = img_name + '.json'
        ann_path = os.path.join(dataset, 'ann', ann_name)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        masks = []
        # if len(ann['objects']) == 0: # Если объектов нет пусть маска будет пустая, не над их выкидывать
        #     continue
        for i in range(len(ann['objects'])):
            if ann['objects'][i]['geometryType'] == 'polygon':
                masks.append(ann['objects'][i])
                continue
        img_path = os.path.join(dataset, 'img', img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_h = img.shape[0]
        orig_w = img.shape[1]
        scale_h = 256 / orig_h
        scale_w = 256 / orig_w
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img_path = os.path.join('images', img_name)
        cv2.imwrite(img_path, img)
        if len(masks) > 0:
            np_masks = [np.zeros((256, 256), dtype=np.uint8)]*2

            for object in ann['objects']: # Если объектов нет весь цикл просто пропустит и маски будут пустые
                if object['classTitle'] == 'parking_side' or object['classTitle'] == 'parking_marked':
                    poly = []
                    for point in object['points']['exterior']:
                        point = [int(point[0]) * scale_w, int(point[1]) * scale_h]
                        poly.append(point)
                    mask_idx = classes.index(object['classTitle'])
                    cv2.fillPoly(np_masks[mask_idx], [np.array(poly, np.int32)], (255, 255, 255), cv2.LINE_AA)
                    
            np_masks = np.dstack(np_masks)
            print(np_masks.shape)
            name, _ = os.path.splitext(img_name)
            ex_path = os.path.join('./masks', name + '.npy')
            np.save(ex_path, np_masks)
            seg_row = (img_name, name + '.npy')
            seg_anns.append(seg_row)
with open('seg_anns.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(seg_anns)

with open('seg_anns.csv', 'r') as f1, open('seg_anns_copy.csv', 'w') as f2:
    f2.writelines(line for line in f1 if line.strip())
