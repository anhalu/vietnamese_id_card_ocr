from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor 
import numpy as np 

import pytesseract 
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

names_index = {0 : 'id', 1 : 'name', 2 : 'birth' } 
imgWidth = 640 
imgHeight = 480
# YOLO 
sources_model = './model/finalTextDetection.pt'
model = YOLO(sources_model) 

# VietOcr 
config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = './model/transformerocr.pth' 

config['device'] = 'cuda:0'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False

detector = Predictor(config)



def ocr(crop_img) : 
    text = detector.predict(crop_img) 
    return text 

def get_text(img): 
    
    results = model.predict(source = img) 
    
    dic = {'id'   : [],  
           'name' : [], 
           'birth': []} 
    
    for box in results[0].boxes : 
        name = names_index[int(box.cls[0])] 
        dic[name].append(box.xyxy[0].cpu().numpy().astype(int)) 
    
    res = {'id'   : '',  
           'name' : '', 
           'birth': ''} 

    for key in dic : 
        for value in dic[key] : 
            crop_img = img.crop(value) 
            
            res[key] = ocr(crop_img)  
    # res['id'] = pytesseract.image_to_string()
    return res 
    
# if __name__ == '__main__':
#     path_image = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/Data_TextDetection/images/train/10.jpg'
     
#     get_text(path_image)
