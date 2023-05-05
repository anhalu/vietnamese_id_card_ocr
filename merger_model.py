from detect_corner import get_transform 
from detect_text import get_text 
from PIL import Image 
import json 
import time 
import argparse 

parser = argparse.ArgumentParser(description = "Enter link to the image") 
parser.add_argument('-l', '--link', required=True, help = "Enter a link or a path") 

args = parser.parse_args() 

link = args.link 

time_start = time.time() 
cropImage = get_transform(link) 
cropImage = Image.fromarray(cropImage) 
resutls = get_text(cropImage) 
time_end = time.time() 


print("time run : " + str(time_end - time_start))
print(resutls)

# with open("my_dict.json", "w", encoding="utf-8") as f:
#     json.dump(my_dict, f, ensure_ascii=False)

with open("results.json", "w", encoding="utf-8") as outfile:
    json.dump(resutls, outfile, ensure_ascii=False, indent=3)

 

