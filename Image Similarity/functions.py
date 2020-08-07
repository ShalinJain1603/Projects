import os
from PIL import Image
import numpy as np
from sklearn.externals import joblib 
model = joblib.load('bullet_firearm.pkl')  

def load_images(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        names.append(filename)
        img = Image.open(os.path.join(folder, filename))
        img=  img.convert('RGB')
        img = img.resize((150,150))
        img = np.array(img)
        if img is not None:
            images.append(img)
    return images,names


def final_answer(data):
    predictions = model.predict(data)
    max_value = np.argmax(predictions,axis = 1)
    final_answer = []
    for i in range(len(max_value)):
        if max_value[i] == 0:
            final_answer.append(11)
        elif max_value[i] == 1:
            final_answer.append(12)
        elif max_value[i] == 2:
            final_answer.append(21)
        elif max_value[i] == 3:
            final_answer.append(22)
        elif max_value[i] == 4:
            final_answer.append(31)
        else:
            final_answer.append(32)
    return final_answer
    