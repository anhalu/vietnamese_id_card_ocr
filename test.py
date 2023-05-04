from detect_corner import get_transform 
import matplotlib.pyplot as plt 


path = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/Data_Corner/images/train/33.jpg' 
img = get_transform(path) 
plt.imshow(img) 
plt.show() 
