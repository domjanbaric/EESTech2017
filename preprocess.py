from PIL import Image,ImageOps, ImageChops
import os, sys
import numpy as np
from skimage import color, feature

def convert(source='dataset',target='converted'):
   
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im=Image.open(read+file)
        f, e = os.path.splitext(read+file)
        outfile = f[f.rfind('\\')+1:f.__len__()] + ".jpg"
        if e=='.gif':
            i = 0
            mypalette = im.getpalette()
            try:
                while 1:
                    im.putpalette(mypalette)
                    new_im = Image.new("RGBA", im.size)
                    new_im.paste(im)
                    background = Image.new("RGB", new_im.size, (255, 255, 255))
                    background.paste(new_im, mask=new_im.split()[3]) # 3 is the alpha channel
                    outfile = f[f.rfind('\\')+1:f.__len__()] +str(i)+".jpg"
                    background.save(write+outfile)
                    i += 1
                    im.seek(im.tell() + 1)
            except EOFError:
                continue
        elif im.mode == "RGBA":
            background = Image.new("RGB", im.size, (255, 255, 255))
            background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
            im=background
        elif im.mode == 'CMYK':
            im = im.convert('RGB')
        elif im.mode == 'L':
            im=np.array(im)
            R=G=B=im
            im=np.dstack((R,G,B))
            print(file)
            im=Image.fromarray(im)
        try:
            im.save(write+outfile)
        except IOError:
            print("cannot convert", read+file)

def resize(width=150,height=150,source='converted',target='resized'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im=im.resize((width,height))
            im.save(write+file)
        except IOError:
            print("cannot resize",read+file)

def grayscale(source='converted',target='black'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im=im.convert("L")
            im.save(write+file)
        except IOError:
            print("cannot resize",read+file)
            
def trim1(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
def trim2(im):
    width, height = im.size
    bg = Image.new(im.mode, im.size, im.getpixel((width-1,height-1)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def croping(source='converted',target='croped'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im=trim1(im)
            im=trim2(im)
            im.save(write+file)
        except IOError:
            print("cannot resize",read+file)

def augment(source='converted',target='augmented'):
    root=os.getcwd()+'\\'
    read=root+source+'\\'
    write=root+target+'\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            out=im.transpose(Image.FLIP_LEFT_RIGHT)
            out.save(write+file+"_mirror")
        except IOError:
            print("Error with augmentation",read+file+"_mirror")


def detect_edge(path,s=2):
    im = Image.open(path)
    try:
        img_matrix = np.array(im)
        img_gray = color.rgb2gray(img_matrix)
        edge = feature.canny(img_gray,sigma=s)
        img_matrix[edge]=0
    except IOError:
        print("Error with detecting edges",path)


def thumb(width=150, height=150, source='converted', target='resized'):
    root = os.getcwd() + '\\'
    read = root + source + '\\'
    write = root + target + '\\'
    if not os.path.exists(write):
        os.makedirs(write)
    listing = os.listdir(read)
    for file in listing:
        im = Image.open(read + file)
        try:
            im.thumbnail((width, height), Image.ANTIALIAS)
            im.save(write + file)
        except IOError:
            print("cannot resize", read + file)
