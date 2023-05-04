from PIL import Image, ImageFilter, ImageEnhance 
import random 
import os 
import shutil



def gaussian_blur(img):
    new_img = img.filter(ImageFilter.GaussianBlur(radius=3))  
    return new_img 


def change_brightness(img, factor) : 
    enhancer = ImageEnhance.Brightness(img)
    new_img = enhancer.enhance(factor=factor)
    return new_img

def create_mosaic_image(img) : 
    original_image = img
    tile_size = 20
    num_tiles_x = original_image.width // tile_size
    num_tiles_y = original_image.height // tile_size 

    mosaic_image = Image.new("RGB", (original_image.width, original_image.height))

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            tile_x = x * tile_size
            tile_y = y * tile_size
            tile_box = (tile_x, tile_y, tile_x + tile_size, tile_y + tile_size)
            tile_image = original_image.crop(tile_box)
            tile_image = tile_image.resize((tile_size, tile_size))
            mosaic_box = (tile_x, tile_y, tile_x + tile_size, tile_y + tile_size)
            mosaic_image.paste(tile_image, mosaic_box)
    return mosaic_image 


def run(sources_img, sources_label) : 
    count = 277
    dir = os.listdir(sources_img)
    
    for item in dir : 
        img = Image.open(sources_img + '/' + item) 

        new_img = gaussian_blur(img) 
        new_img.save(sources_img + '/' + str(count) + ".jpg") 
        shutil.copy(sources_label + '/' + item[:-4] + '.txt', sources_label + '/' + str(count) + '.txt' ) 
        count += 1
        
        rand = random.uniform(1, 1.3)
        new_img = change_brightness(new_img, rand)
        new_img.save(sources_img + '/' + str(count) + ".jpg") 
        shutil.copy(sources_label + '/' + item[:-4] + '.txt', sources_label + '/' + str(count) + '.txt' ) 
        count += 1 
        
        new_img = create_mosaic_image(new_img) 
        new_img.save(sources_img + '/' + str(count) + ".jpg") 
        shutil.copy(sources_label + '/' + item[:-4] + '.txt', sources_label + '/' + str(count) + '.txt' ) 
        count += 1 
        
        
    


if __name__ == "__main__": 
    sources_img   = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/finalDataset/images/train'
    sources_label = '/home/anhalu/anhalu-data/AN.LAB/id_card_ocr/Data/finalDataset/labels/train' 
    
    run(sources_img, sources_label) 

