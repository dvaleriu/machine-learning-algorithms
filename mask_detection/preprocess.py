import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
0. Transformare in grayscale pt ca avem mai multe tiprui de masca
1. Reducere rezolutie la {rezX, rezY}
2. Normalizare 0-1

"""
if __name__ == "__main__":

    #inputs
    dataset_path = "./Data"
    output_path = "./Data/preprocesat"
    desired_rezolution = (64,64)

    #daca nu exista folderul
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path, class_folder)

        #iterare imagini din foldere
        for file in os.listdir(class_folder_path):
            file_full_path = os.path.join(class_folder_path,file)

            #citire imagine + 0 pt grayscale
            img = cv2.imread(file_full_path, 0) 

            #resize
            img_resized = cv2.resize(img, desired_rezolution)
           
            #normalizare
            norm_img = (img_resized - np.min(img_resized))/(np.max(img_resized ) - np.min(img_resized))

            #salvare
            gr = 0 if class_folder == "nu_masca" else (1 if class_folder == "masca_prost" else 2)
            output_path_save = os.path.join(output_path, file.replace(".png", ".npz"))
            np.savez(output_path_save, img =norm_img, gr= gr)

            print(f"Salvat la {output_path_save}")





    
