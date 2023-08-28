# CLANNAD Face Recognition

日本語の説明：https://github.com/tomkaoru/CLANNAD-Face-Recognition/blob/main/README_ja.md

**Description：**

A program that detects the face(s) of character(s) from the anime CLANNAD and outputs the name of the character(s) and their location in the image. The reason I choose CLANNAD for face recognition is that the characters from CLANNAD have similar eye shapes and I thought it is difficult to correctly recognise each character compared to characters in other animes.  The program can recognise 8 characters mentioned on CLANNAD's [official homepage](https://www.tbs.co.jp/clannad/clannad1/04chara/chara.html).

  | Input image | Output image |
  | ------ | ------ |
  | ![くらなど](https://user-images.githubusercontent.com/52717342/175796835-4bb0178b-7631-4cb9-b43f-c0a9670c3118.jpeg) | ![f2c533a00ecb95ead668ed36a7c42963b3d02ce64691bf10c2b779dd](https://user-images.githubusercontent.com/52717342/175796837-a6d859ba-1072-410f-a5b4-cf936ebfec58.jpeg) |
  | © VisualArt's/Key/光坂高校演劇部 ／Source: [RENOTE](https://renote.jp/articles/11191) |  |




**Files：**

- ```lbpcascade_animeface.xml``` - [Cascade file](https://github.com/nagadomi/lbpcascade_animeface) used to detect anime faces.
- ```face.py``` - Uses ```lbpcascade_animeface.xml``` to detect faces from images and save the facial images.
- ```da.py``` - increases the number of training images by creating images of different blur levels and brightness. 
- ```train.py``` - Creates the model and tests it on test data.
- ```model_200_160_80_epoch30_30.h5``` - Model to recognise CLANNAD character from facial images.
- ```test.py``` - Takes in an image and outputs the name of the character(s) and their location in the image.
- ```app.py``` - Uses Streamlit to make a web app.
  - Link to the app: https://rtorii-clannad-face-recognition-app-95w7hy.streamlitapp.com/


# What I did step by step

    1. Data (facial images of each character) collection and classification.
    2. Increase the amount of train data (data augmentation).
    3. Create a model.
    4. Test the model. 
    5. Create a program to take in an image and output the name of the character(s) and their location in the image.
    6. Use Streamlit to convert the program into a web app.

1 .  **Data (facial images of each character) collection and classification.```face.py```**
- Collected images from CLANNAD by taking screenshots on episodes (from episode 1 to episode 22). Used ``lbpcascade_animeface.xml``` to detect faces from the screenshots and save the facial images. 
- Classified all facial images manually and saved the images with the same character in the same folder.
- Used Python library, ```split-folder```, to divide the images into train set and test set.

    |  Character  |  # of images collected（train set : test set）   |  Character   |# of images collected（train set : test set）    |
    | --- | --- | --- | --- | 
    |   Fuko Ibuki  |   133 (106 : 27)  |  Kotomi Ichinose  | 127 (101 : 26)   |
    |   Kyou Fujibayashi  |   96 (76 : 20)  |   Nagisa Furukawa  |  229 (184 : 45)   |
    |   Ryou Fujibayashi  |  88 (70 : 18)   |  Tomoya Okazaki   |  286 (231 : 55)   | 
    |   Tomoyo Sakagami |  128 (102 : 26)  |  Youhei Sunohara   |  119 (95 : 24)  |

    ※ Although Youhei Sunohara has scenes in CLANNAD ～After Story～ where he has dark hair, I used images of him with blonde hair in this project since his hair is blonde in CLANNAD (from episode 1 to episode 22).

2 . **Increase the amount of train data (data augmentation).　```da.py```**
- For each image in the train set, the program created 3 images with different blur levels. In addition, it created 5 images with different brightness after creating 3 images with different blur levels. The number of images in the train set is increased by 3 x 5 = 15 times its original.
- The total number of training images for the eight characters became 14,475.

    |  Character   |  # of training images    |  Character   | # of training images     |
    | --- | --- | --- | --- | 
    |   Fuko Ibuki  |   1590  |  Kotomi Ichinose   | 1515    |
    |   Kyou Fujibayashi  |   1140  |   Nagisa Furukawa  |  2760   |
    |   Ryou Fujibayashi  |  1050   |  Tomoya Okazaki   |  3465   | 
    |   Tomoyo Sakagami  |  1530   |  Youhei Sunohara   |  1425   |


3 . **Create a model. ```train.py```**
- Created a sequential model (four-layer neural network) using Keras. Then trained the model on training images resized to 100x100. 
- Model takes in RGB information from all pixels (100x100x3 = 30000). It has the output layer with 8 nodes (for 8 characters). The character with the highest output value is the one the model recognises as the character in the image.
- Set the number of epochs to 30 (to prevent overfitting).

    ```
    Epoch 21/30
    483/483 [==============================] - 6s 12ms/step - loss: 0.1139 - accuracy: 0.9727
    Epoch 22/30
    483/483 [==============================] - 6s 12ms/step - loss: 0.1058 - accuracy: 0.9719
    Epoch 23/30
    483/483 [==============================] - 7s 14ms/step - loss: 0.0948 - accuracy: 0.9771
    Epoch 24/30
    483/483 [==============================] - 7s 14ms/step - loss: 0.0843 - accuracy: 0.9800
    Epoch 25/30
    483/483 [==============================] - 6s 13ms/step - loss: 0.0765 - accuracy: 0.9825
    Epoch 26/30
    483/483 [==============================] - 6s 12ms/step - loss: 0.0761 - accuracy: 0.9813
    Epoch 27/30
    483/483 [==============================] - 6s 12ms/step - loss: 0.0686 - accuracy: 0.9844
    Epoch 28/30
    483/483 [==============================] - 6s 12ms/step - loss: 0.0636 - accuracy: 0.9843
    Epoch 29/30
    483/483 [==============================] - 6s 13ms/step - loss: 0.0574 - accuracy: 0.9869
    Epoch 30/30
    483/483 [==============================] - 6s 13ms/step - loss: 0.0513 - accuracy: 0.9882
    ```
4 . **Test the model. ```train.py```**
- The accuracy of the model tested on the test set is 226 / 241 = 93.8%.
- The model correctly recognised Tomoya Okazaki, who had the most training images, on all of his test images. 
- The model did not do well on test images of Kyou and Ryou Fujibayashi (twin sisters) who have the same hair color and have less training images compared to other characters.


  |  Character   |   # of correctly predicted images  |  # of test images   | Accuracy   | Character   |  # of correctly predicted images   |  # of test images   | Accuracy   |
  | --- | --- | --- | --- | --- | --- | --- | --- | 
  |   Fuko Ibuki  |  26   |   27  |  96.3%   |  Kotomi Ichinose   |  24   | 26   |  92.3%  | 
  |  Kyou Fujibayashi   |   17  |  20   |   85%  |  Nagisa Furukawa   |  43   |  45   |  95.6%   | 
  |  Ryou Fujibayashi   |  15   |   18  |   83.3%  |   Tomoya Okazaki  |  55   |  55   |   100%    | 
  |  Tomoyo Sakagami   |  23   |  26   |  88.5%  |  Youhei Sunohara  |  23   |  24   |  95.8%   | 

5 . **Create a program to take in an image and output the name of the character(s) and their location in the image.```test.py```**
  
  The program outputs an image showing the name and location of the character on a white background. I originally wanted to modify the input image, i.e. I wanted put an square on each face detected. But I choose not to, because I saw articles stating that modifying anime images and posting the modified images on the internet infringes copyright. 

**What the program does step by step：**
1. Uses ```lbpcascade_animeface.xml``` to detect face(s) in input image/
2. Gives each facial image to the model and recognises the character with the highest output value as the one in the image.
3. Outputs an image showing the name and location of the character(s) on a white background.

  | Input image | Output image |
  | ------ | ------ |
  | ![5aca841a5e7f9cb1fa2fe36d74f51c486e17d7d2d0715424effb70ab](https://user-images.githubusercontent.com/52717342/175799918-d6354880-c3ba-4f1b-829c-94e07b80b43e.jpeg) | ![f45514849387e57cca1283ba6311a07841285ef809d10fd42e28842f](https://user-images.githubusercontent.com/52717342/175799923-161d704f-fdc0-4945-ad04-cc6cd7823cb7.jpeg) |
  | © VisualArt's/Key/光坂高校演劇部 ／Source: [アニメミル](https://animemiru.jp/articles/7675/) |  |

- Recognised Nagisa Furukawa who is closing her eyes and the crying Youhei Sunohara. Tomoyo Sakagami's face is not detected by ```lbpcascade_animeface.xml``` because she is in profile。

| Input image | Output image |
| ------ | ------ |
| ![25a02d32019f1987c8fa9a83b159cfa6b891afbe2346d2d3122d07b10cc1766f _SX1080_](https://user-images.githubusercontent.com/52717342/175796378-d3f18259-b42b-45ee-ae19-a43b3e3724a6.jpg) | ![696b9da8dc6eac7d29d7a1c5d4db7dd0f19f2c69e7944f40a04813ec](https://user-images.githubusercontent.com/52717342/175796921-2646e6b2-0125-444b-9e34-ff99aa62e089.jpeg) |
| © VisualArt's/Key/光坂高校演劇部 ／Source: [Amazon](https://www.amazon.co.jp/CLANNAD-AFTER-STORY%E3%80%90TBS%E3%82%AA%E3%83%B3%E3%83%87%E3%83%9E%E3%83%B3%E3%83%89%E3%80%91/dp/B00FYKXTGS) |  |


6 . **Use Streamlit to convert the program into a web app.**
- Link to the web app: https://rtorii-clannad-face-recognition-app-95w7hy.streamlitapp.com/

**How to use it:**
1.  Upload an image with the character(s) from CLANNAD on it.
2.  Press the ```Process``` Button to see the output.

| Homepage | 
| ------ |
| <img width="1439" alt="Screen Shot 2022-06-26 at 13 20 10" src="https://user-images.githubusercontent.com/52717342/175799260-cb8ab7e3-ba52-4e46-919c-396fdcfbf745.png"> |


Created by Ryota Torii <rtorii@protonmail.com> on 06/26/22
