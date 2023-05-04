from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor 
import numpy as np 
from PIL import Image 
import cv2 
import os 


names_index = ['top left', 'top right', 'bottom left', 'bottom right'] 
imgWidth = 640 
imgHeight = 480


sources_model = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/model/model_detect_corner.pt' 
model = YOLO(sources_model) 


def get_index_one_point(coordinates) : 
    return [int((coordinates[0][0] + coordinates[0][2])//2) , int((coordinates[0][1] + coordinates[0][3])//2)]

def changesize(img) :
    w, h, d = img.shape
    ratio = w/h 
    newW = 640 
    newH = int(newW/ratio)
    img = cv2.resize(img, (newW, newH), interpolation = cv2.INTER_AREA)
    return img 

def get_transform(path_image) : 
    
    img = cv2.imread(path_image) 
    
    img = changesize(img) 
    
    results = model.predict(source = img) 

    dic = {'top left' : [], 
           'top right' : [], 
           'bottom left' : [], 
           'bottom right' : []}

    for box in results[0].boxes :
        name = names_index[int(box.cls[0])]
        if (len(dic[name]) == 0) : 
            dic[name] = get_index_one_point(box.xyxy)
    
    for key in dic : 
        if len(dic[key]) == 0 : 
            print("Ảnh đầu vào không hợp lệ !") 
            exit() 
    
    dic['top left'] = dic['top left'][0] - 10, dic['top left'][1] - 10  
    dic['top right'] = dic['top right'][0] + 10, dic['top right'][1] - 10 
    dic['bottom left'] = dic['bottom left'][0] - 10, dic['bottom left'][1] + 10 
    dic['bottom right'] = dic['bottom right'][0] + 10, dic['bottom right'][1] + 10 
     
    
            
    sources_point = np.float32([dic['top left'], dic['top right'], dic['bottom right'], dic['bottom left']])
    dest_points = np.float32([[0,0], [imgWidth, 0], [imgWidth, imgHeight], [0, imgHeight]]) 
    matrix = cv2.getPerspectiveTransform(sources_point, dest_points) 
    crop_img = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
    
    return crop_img 

    # print(path_image)
    # os.remove(path_image) 
    # cv2.imwrite(path_image , crop_img) 




# if __name__ == '__main__': 
#     path_image = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/Data_TextDetection/Images/train'
#     dirs = os.listdir(path_image)
#     for dir in dirs : 
#         get_transform(os.path.join(path_image, dir), sources_model) 


