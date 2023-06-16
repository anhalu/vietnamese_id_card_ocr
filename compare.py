import json 
import os
import argparse 

parser = argparse.ArgumentParser(description = "Enter a path to the image") 
parser.add_argument('-l', '--label', required=True, help = "Enter a path to label")  
parser.add_argument('-p', '--predict', required=True, help = "Enter a path to predict") 

args = parser.parse_args() 

labelPath = args.label 

predictPath = args.predict

count = 0 
total = os.listdir(labelPath)
total = sorted(total, key= lambda x: int(x.split('.')[0])) 

for item in total :  
    path = os.path.join(predictPath, item)
    if os.path.isfile(path):
        with open(path, 'r') as f : 
            predict = json.load(f) 
        
        with open(os.path.join(labelPath, item), 'r') as f : 
            label = json.load(f) 

        # print(label, predict, end='\n') 
        print(item, end=' ')
        if label['birth'] == predict['birth'] : 
            count += 1 
            print("Dung") 
        else : 
            print("Sai")
        print(label['birth'])
        print(predict['birth']) 
        print('\n')
print(f"Du doan dung : {count} / {len(total)}")
print( "Acc : {:.2f} %".format(count/len(total)*100))
        

