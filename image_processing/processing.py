from PIL import Image
import os 
import random 
sources = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/bomoi/images/train'



for dir in os.listdir(sources) : 
    img = Image.open(os.path.join(sources, dir)) 
    img = img.convert('RGB')
    
    w, h = img.size
    ratio = w/h 
    newW = 640 
    newH = int(newW/ratio)
    img = img.resize((newW, newH), Image.LANCZOS) 
    img.save(sources + '/' +  dir[:-4]+ '.jpg') 
    
    # os.remove(os.path.join(sources, dir))
    
    


