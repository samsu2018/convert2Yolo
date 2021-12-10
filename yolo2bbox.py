# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:00:10 2021

@author: Sam

Function: Convert 'yolo format' to 'bounded box format' for calculate mAP

reference: https://blog.csdn.net/weixin_43508499/article/details/118600392
"""
import glob
import os
import cv2

# classname
cls_file = './coco/coco.txt'

with open(cls_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

def get_basename(filepath):
    '''
        filename and extension name
        filepath = '/home/ubuntu/python/example.py'
        return basename = example.py
    '''
    return os.path.basename(filepath)

def get_filename_only(filepath):
    '''filepath = '/home/ubuntu/python/example.py'
       return filename = example
    '''
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]  

def created_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

def yolo_to_bbox(temp, w, h):
    '''
    imput yolo format and img' weight and height
    return retangle format x1, x2, y1, y2
    '''
    x_, y_, w_, h_=eval(temp[1]), eval(temp[2]), eval(temp[3]), eval(temp[4])
    x1 = w * x_ - 0.5 * w * w_
    x2 = w * x_ + 0.5 * w * w_
    y1 = h * y_ - 0.5 * h * h_
    y2 = h * y_ + 0.5 * h * h_
    return x1, y1, x2, y2

def join_label_name(label):
    label_name = label.split()
    if len(label_name)>1:
        label = '_'.join(label_name)

    return label
    
def single(f, out_path, yoloPath, imgPath):
    '''imput image file path, for get images w,h'''
    print(f)
    img = cv2.imread(f)
    h, w, channels = img.shape
    img_dir = imgPath.split('/')[-1]   # get images folder name
    yolo_dir = yoloPath.split('/')[-1] # get yolo folder name
    f = f.replace(img_dir, yolo_dir)
    f_yolo = f.replace(".jpg", ".txt")
    print(f_yolo)
    if os.path.isfile(f_yolo):
        with open(f_yolo, 'r') as f:
            lines = f.readlines()
    else:
        # skip
        return

    # Write retcangle txt 
    basename = get_basename(f_yolo)
    bbox_file = os.path.join(out_path, basename)
    print(bbox_file)
    file = open(bbox_file,'w')
    for line in lines:
        temp = line.split()
        label = join_label_name(classes[int(temp[0])])
        # ['1', , '0.43906', '0.52083', '0.34687', '0.15']
        x1, y1, x2, y2 = yolo_to_bbox(temp, w, h)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print('{} {} {} {} {}'.format(label, x1, y1, x1, y2))
        file.write('{} {} {} {} {}\n'.format(label, x1, y1, y2, y2))

#        # ===Draw bounding box===
#        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0))
#    file.close()
#    print('write to retange.txt')


if __name__ == '__main__':
    imgPath = './coco/train2014images' 
    yoloPath = './coco/train2014yolo'
    output_dir = './coco/train2014bbox'
    created_directory(output_dir)

    # === Single or multiple DEMO ===
    multiple = 1
    if multiple:
        for f in glob.glob(os.path.join(imgPath, "*.jpg")):
            single(f, output_dir, yoloPath, imgPath)
    else:
        f = './coco/val2014/COCO_val2014_000000000395.jpg'
        if os.path.isfile(f):
            single(f, output_dir, yoloPath, imgPath)
            print('Save to file')
