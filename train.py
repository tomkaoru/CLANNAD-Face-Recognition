from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
import os


def main():
    all_imgs, all_labels = load_imgs()
    all_imgs = np.array(all_imgs)
    Y = to_categorical(all_labels, 8)
    model = create_model()
    model.fit(all_imgs, Y, epochs = 30, batch_size = 30)# train
    test(model)
    model.save("model_200_160_80_epoch30_30.h5")

def create_model(): # create a neural network model
    model = Sequential()
    model.add(Dense(200, activation = 'relu', input_dim = 30000))#input_dim is 3(rgb)*100*100
    model.add(Dropout(0.2))
    model.add(Dense(160, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(80, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation = "softmax"))
    model.compile(loss = "categorical_crossentropy", optimizer = Adam(learning_rate=1e-5), metrics = ["accuracy"])
    return model


def load_imgs():
    all_imgs = [] #train images
    all_labels = [] #lables for train sets
    for t_img in os.listdir('train_data/'):
        if t_img is None or t_img == ".DS_Store":
            continue
        t_img_name,t_img_ex = os.path.splitext(t_img)
        if t_img_name[:4] == "fuko":
            t_img_label = 0
        elif t_img_name[:4] == "koto":
            t_img_label = 1
        elif t_img_name[:4] == "kyou":
            t_img_label = 2
        elif t_img_name[:4] == "nagi":
            t_img_label = 3
        elif t_img_name[:4] == "ryou":
            t_img_label = 4
        elif t_img_name[:4] == "youh":
            t_img_label = 5
        else:
            if t_img_name[5] == "a":#tomoya
                t_img_label = 6
            else:#tomoyo
                t_img_label = 7
        all_labels.append(t_img_label)
        loaded_img = np.array(Image.open("train_data/" + t_img).resize((100, 100)))
        loaded_img = loaded_img.transpose(2, 0, 1) # change order to rgb
        loaded_img = loaded_img.reshape(1, loaded_img.shape[0] * loaded_img.shape[1] * loaded_img.shape[2]).astype("float32")[0]
        all_imgs.append(loaded_img / 255)
    return all_imgs, all_labels

def test(model): # test the model on test dataset.
    total = [0 for i in range(8)]
    correct = [0 for i in range(8)]
    for test_dir in os.listdir("test"):
        if test_dir == ".DS_Store":
            continue
        if test_dir == "fuko":
            test_label = 0
        elif test_dir == "kotomi":
            test_label = 1
        elif test_dir == "kyou":
            test_label = 2
        elif test_dir == "nagisa":
            test_label = 3
        elif test_dir == "ryou":
            test_label = 4
        elif test_dir == "tomoya":
            test_label = 6
        elif test_dir == "tomoyo":
            test_label = 7
        else: #youhei
            test_label = 5

        for test_img in os.listdir("test/"+test_dir):
            if test_img is None or test_img == ".DS_Store":
                continue
            test_img = np.array(Image.open("test/"+test_dir+'/'+test_img).resize((100, 100)))
            test_img = test_img.transpose(2, 0, 1)
            test_img = test_img.reshape(1, test_img.shape[0] * test_img.shape[1] * test_img.shape[2]).astype("float32")[0]
            # result = model.predict(np.array([test_img / 255.]))
            result = np.argmax(model.predict(np.array([test_img / 255])), axis=-1)
            if result == test_label:
                correct[test_label]+=1
            total[test_label]+=1
    for i in range(8):
        print(correct[i],total[i])

if __name__ == '__main__':
    main()
