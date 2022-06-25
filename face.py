import cv2
import os

# reads all screenshots from 'CLANNAD' folder which contains 22 folders (holds screenshots from one episode)
# e.g, folder '1' contains screenshots from episode 1

def main():
    anime_cascade = cv2.CascadeClassifier('lbpcascade_animeface.xml') #lbpcascade_animeface.xml is in the same directory as face.py
    folder_dir = [str(i) for i in range(1,23)]
    for i in folder_dir:# for each folder
        img_path = "CLANNAD/" + i
        for screenshot  in os.listdir(img_path): #for each screenshot in the folder
            loaded_img= cv2.imread(img_path + "/" + screenshot), 1)
            if loaded_img is None:
                continue
            img_gray=cv2.cvtColor(loaded_img, cv2.COLOR_BGR2GRAY)
            faces=anime_cascade.detectMultiScale(img_gray, scaleFactor=1.08, minNeighbors=1, minSize=(10,10))
            new_dir = "f_"+i
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    face = loaded_img[y:y+h, x:x+w]
                    file_name = os.path.join(new_dir, screenshot)
                    cv2.imwrite(file_name, face)

if __name__ == '__main__':
    main()
