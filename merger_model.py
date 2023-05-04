from detect_corner import get_transform 
from detect_text import get_text 
from PIL import Image 
import json 
import time 

path_image = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/3.jpg' 


time_start = time.time() 
cropImage = get_transform(path_image) 
cropImage = Image.fromarray(cropImage) 
resutls = get_text(cropImage) 
time_end = time.time() 

print("time run : " + str(time_end - time_start))
print(resutls)

# with open("my_dict.json", "w", encoding="utf-8") as f:
#     json.dump(my_dict, f, ensure_ascii=False)

with open("results.json", "w", encoding="utf-8") as outfile:
    json.dump(resutls, outfile, ensure_ascii=False)

 

