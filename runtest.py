from detect_corner import get_transform 
from detect_text import get_text 
from PIL import Image 
import json 
import os
import argparse 

parser = argparse.ArgumentParser(description = "Enter link to the image") 
parser.add_argument('-l', '--link', required=True, help = "Enter a link or a path") 

args = parser.parse_args() 

link = args.link 

if not os.path.exists('./predicts') : 
    os.mkdir('./predicts')
else : 
    for dir in os.listdir('./predicts') : 
        os.remove(os.path.join('./predicts', dir ))


dirs = os.listdir(link)

for dir in dirs : 
    with open(f'./predicts/{dir[:-4]}.json', 'w', encoding='utf-8') as outfile : 
        path = os.path.join(link, dir)
        cropImage = get_transform(path) 
        cropImage = Image.fromarray(cropImage) 
        resutls = get_text(cropImage, path) 
        json.dump(resutls, outfile, ensure_ascii= False) 
    
# with open("my_dict.json", "w", encoding="utf-8") as f:
#     json.dump(my_dict, f, ensure_ascii=False)




