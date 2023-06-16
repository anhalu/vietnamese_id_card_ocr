import cv2
import numpy as np
import os 
import random 
import pybboxes

def bb2yolo(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def yolo2bb(size, box) : 
    x_center = box[0] * size[0]
    y_center = box[1] * size[1] 
    w = box[2] * size[0] 
    h = box[3] * size[1] 
    return x_center, y_center, w, h


def get_coordinates_after_resize(oldx, oldy, oldsize, newsize) : 
    
    newx = oldx * (newsize[0] / oldsize[0])
    newy = oldy * (newsize[1] / oldsize[1])
    return int(newx), int(newy)  

def changesize(img) :
    h, w = img.shape[0], img.shape[1] 
    ratio = w/h 
    newW = 640 
    newH = int(newW/ratio)
    img1 = cv2.resize(img, (newW, newH), cv2.INTER_AREA) 
    return img1

def get_rotate_image(list_center, img) : 
    
    angle = random.randint(-30, 30) 
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)

    cos_theta = np.abs(M[0, 0])
    sin_theta = np.abs(M[0, 1])
    new_w = int(h * sin_theta + w * cos_theta)
    new_h = int(h * cos_theta + w * sin_theta)

    M[0, 2] += (new_w / 2) - w/2
    M[1, 2] += (new_h / 2) - h/2

    rotated_image = cv2.warpAffine(img, M, (new_w, new_h))
    rotated_image = changesize(rotated_image) 
    
    list_res = [] 
    
    for point in list_center : 
        pt = (point[1], point[2])
        # print(pt)
        pt_homog = np.hstack((pt, 1)) 
        new_pt_homog = np.dot(M, pt_homog)
        # newx, newy = new_pt_homog[0], new_pt_homog[1]
        newx, newy = get_coordinates_after_resize(new_pt_homog[0], new_pt_homog[1], (new_w, new_h), (rotated_image.shape[1], rotated_image.shape[0]))        

        list_res.append([int(point[0]), int(newx), int(newy)])
        # cv2.circle(rotated_image, center=(int(newx), int(newy)), color=(0,0,255), radius=5, thickness=-1)
    
    # cv2.imshow('rotate', rotated_image)
    # cv2.waitKey(10000) 
    return list_res, rotated_image 

def main(imagepath, filepath, count) : 
    
    img = cv2.imread(imagepath) 
    H, W = img.shape[:2]
    # print(H, W)
    f = open(filepath, 'r') 
    data = f.readlines() 
    
    
    list_center = []
    for dt in data : 
        if len(dt) == 0 : 
            continue  
        
        dt = dt.split(' ')
        dt = [float(x) for x in dt] 
        
        bb = yolo2bb((W, H), dt[1:])
        # t = pbx.convert_bbox((dt[1], dt[2], dt[3], dt[4]), from_type='yolo', to_type='voc', image_size= (W, H))
        # print(bb)
        list_center.append([dt[0], bb[0], bb[1]]) 

    list_res, rotated_image = get_rotate_image(list_center , img) 
    
    path = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/new/images/train/'  + str(count) + '.jpg'
 
    
    cv2.imwrite(path , rotated_image) 
    
    newH, newW,_ = rotated_image.shape 
    
    # print(newH, newW) 
    
    # for point in list_res : 
    #     print(point)
    #     cv2.circle(rotated_image, (point[1], point[2]), radius=5, color = (255, 0, 0), thickness=-1)
    
    
    # print(rotated_image.shape) 
    
    flag = True
    
    with open(f'/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/new/labels/train/{count}.txt', 'w') as file : 
        for point in list_res : 
            # print(point)
            if point[1] > newW or point[1] <0 or point[2] > newH or point[2] < 0 : 
                
                continue  
            # pt = bb2yolo((W,H), (point[1] - 5, point[2] - 5, point[1] + 5, point[2] + 5))
            
            flag = False 
            pt = pybboxes.convert_bbox((point[1] - 15, point[2] - 15, point[1] + 15, point[2] + 15), from_type = 'voc', to_type = 'yolo', image_size = (newW, newH))
            
            # cv2.rectangle(rotated_image, (point[1] - 20, point[2] - 20), (point[1] + 20, point[2] + 20), color = (255,0,0), thickness= 1)
            
            file.write(f'{point[0]} {pt[0]} {pt[1]} {pt[2]} {pt[3]}\n')
        
        if flag : 
            print('Error' + str(count)) 
        
    # cv2.imshow('rotated', rotated_image)
    
    # cv2.waitKey(0)
    
    # cv2.imshow('img', img)
    # cv2.waitKey(0)    
    
if __name__ == '__main__': 
    sources = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/new/images/train'
    label   = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/new/labels/train'
    
    count = 433
    for file in range(1, 433):
        imagepath = os.path.join(sources, str(file) + ".jpg" ) 
        filepath  = os.path.join(label, str(file)  + '.txt') 
        
        main(imagepath, filepath, count) 
        count += 1 
        
        