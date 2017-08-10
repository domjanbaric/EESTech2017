# -*- coding: utf-8 -*-
"""
Created on Wed May 10 07:23:57 2017

@author: Ana
"""
import numpy as np
from PIL import Image
from keras.models import load_model
import os
import argparse as ap

model = load_model("boja3.h5")

def test(slija="18_krava.jpg"):
    
    ime=["Adele","Babette","Cecile","Doerte","Elsa","Fabala","Gesa","Helvetia","Isabella","Janette","Kiera","Letitia"]
    root=os.getcwd()+'\\'


    im = Image.open(root + slija)
    im=im.resize((270,180))
    xtest=np.zeros((1,180,270,3),dtype=np.int)
    xtest[0,]=np.array(im)
    
    xtest=xtest.astype('float32')
    xtest/=255
    
    t=model.predict(xtest)
    br=np.argmax(t)
    print(ime[br])
    
if __name__=="__main__":
    parser=ap.ArgumentParser()
    parser.add_argument('-i',"--image",help="Name of picture",required=True)
    args=vars(parser.parse_args())
    image_path=args["image"]
    test(image_path)