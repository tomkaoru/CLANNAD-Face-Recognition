from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import os
import streamlit as st

characters = ["Fuko","Kotomi","Kyou","Nagisa","Ryou","Youhei","Tomoya","Tomoyo"]

colors = [
(0,128,0), #green
(200,170,200),#thistle
(128,0,128),#purple
(200,100,240),#cornsilk
(180,0,180), #fuchsia
(190,190,0),#yellow
(0,0,128), #navy
(200,200,200)# silver
]


def main():
    st.title("CLANNAD Face Detection")
    st.write('This program takes in an image and detects the face(s) of character(s) from the anime CLANNAD. Then it outputs their name and location within the input image.')
    st.write('The program can detect 8 characters: Tomoya Okazaki, Nagisa Furukawa, Kyou Fujibayashi, Ryou Fujibayashi, Tomoyo Sakagami, Fuko Ibuki, Kotomi Ichinose, Youhei Sunohara')
    st.write('Detail of each character: '+'https://www.tbs.co.jp/clannad/clannad1/04chara/chara.html')
    reading_model = st.empty()
    reading_model.text("Please wait while the program loads the model...")
    model,anime_cascade = load_nnmodel_and_cascade()
    reading_model.empty()
    input_img = st.file_uploader("Upload an image that contains a character(s) from CLANNAD", type=['jpg','jpeg','png'])
    pressed = st.button('Process')
    if pressed and input_img is not None:
        process(input_img, model, anime_cascade)

@st.cache(allow_output_mutation=True)
def load_nnmodel_and_cascade():
    model = load_model("model_200_160_80_epoch30_30.h5")
    cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml')
    return model, cascade

def process(input_img, model, anime_cascade):
    input_img = Image.open(input_img).convert('RGB')
    input_img = input_img.save("img.jpg")
    input_img = cv2.imread("img.jpg",1) #  test image
    img_gray=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    blank_img = np.zeros(input_img.shape,dtype=np.uint8)
    blank_img.fill(255)
    faces=anime_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10,10))#
    if len(faces)>0:
        for (x,y,w,h) in faces:
            face = input_img[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_rgb = Image.fromarray(face_rgb)
            face_rgb = np.array(face_rgb.resize((100,100)))
            face_rgb = face_rgb.transpose(2, 0, 1)
            face_rgb = face_rgb.reshape(1, face_rgb.shape[0] * face_rgb.shape[1] * face_rgb.shape[2]).astype("float32")[0]
            result = np.argmax(model.predict(np.array([face_rgb / 255])), axis=-1)
            # print(characters[result[0]])
            cv2.rectangle(blank_img, pt1=(x, y), pt2=(x+w, y+h), color=colors[result[0]], thickness=3, lineType=cv2.LINE_8, shift=0)
            cv2.putText(blank_img, text=characters[result[0]], org=(x+1, y-1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=colors[result[0]], thickness=3, lineType=cv2.LINE_4)
    # cv2.imwrite("output.jpg",blank_img)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    st.image(blank_img, caption='output image', use_column_width='auto')
    st.image(input_img, caption='original image', use_column_width='auto')



if __name__ == '__main__':
    main()
