import cv2
import os
import numpy as np
# from scipy import ndimage

def main():
    folder_dir = ["fuko","kotomi","kyou","nagisa","ryou","tomoya","tomoyo","youhei"]
    new_dir = "train_data"
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for i in folder_dir:# from eposode 1 to the last episode (22nd)
        img_path = "output/train/" + i
        n = 0
        for screenshot in os.listdir(img_path): #for each screenshot in each episode
            if screenshot is None or screenshot == ".DS_Store":
                continue
            loaded_img = cv2.imread(img_path + "/" + screenshot, 1)
            loaded_img = cv2.resize(loaded_img, (100, 100))
            _,extension = os.path.splitext(screenshot)
            if loaded_img is None:
                continue
            for blur_num in [0,3,6]: # for each screenshot, create 3 different blurred images.
                if blur_num != 0:
                    blurred_img = cv2.blur(loaded_img, (blur_num, blur_num))
                else:
                    blurred_img = loaded_img
                # cv2.imshow("blur",blurred_img)
                # cv2.waitKey(0)
                blurred_name = new_dir + "/" + i + str(n) + "_b" + str(blur_num)
                for alpha in [0.84,0.92,1,1.08,1.16]: # for each image, create 5 images with different alpha
                    alpha_img = np.clip(alpha * blurred_img, 0, 255).astype(np.uint8)
                    alpha_name = blurred_name + "_a" + str(alpha)
                    cv2.imwrite(alpha_name + extension, alpha_img) # save the image
            n+=1

if __name__ == '__main__':
    main()
