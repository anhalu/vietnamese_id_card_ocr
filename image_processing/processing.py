from PIL import Image


sources = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/Data_TextDetection/Images/train'


for count in range(23, 24) : 
    img = Image.open(sources +'/'+ str(count) + '.jpg')
    w, h = img.size
    ratio = w/h 
    newW = 640 
    newH = int(newW/ratio)
    img = img.resize((newW, newH), Image.LANCZOS) 
    img.save(sources + '/' + str(count) + '.jpg', quality = 100) 
    


