import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/transformerocr.pth' 

config['device'] = 'cuda:0'
config['cnn']['pretrained']=False
config['predictor']['beamsearch']=False

detector = Predictor(config)

path_img = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/check.png' 
img = Image.open(path_img) 
img = img.convert('L')
plt.imshow(img)
s = detector.predict(img)
print(s)