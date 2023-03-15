"""
using JSON
Datasetpath -> [train:{img1, img2,...}, test:{img....}, validation:{....}]

"""

import os
import random
import json



if __name__ == "__main__":
    #input data
    dataset_path = 'Data'
    split_ratio = [0.7, 0.15, 0.15]


    #dict pentru stocare splituri
    split = {"train":[], "test":[], "val":[]}

    #interare prin foldere
    for class_folder in os.listdir(dataset_path):
        class_folder_path = os.path.join(dataset_path,class_folder)
        print(class_folder_path)

        #toate imaginile din clasa 
        all_images = os.listdir(class_folder_path)
        
        train_idx = int(split_ratio[0] * len(os.listdir(class_folder_path)))
        val_idx = int(split_ratio[1] * len(os.listdir(class_folder_path)))
        test_idx = int(split_ratio[2] * len(os.listdir(class_folder_path)))

        random.shuffle(all_images)


        #eliminare extensii
        all_images = [x.replace(".png", "") for x in all_images]
        
        #split the list
        traintemp = all_images[:train_idx]
        valtemp = all_images[train_idx:train_idx + val_idx]
        testtemp = all_images[train_idx + val_idx:]

        #append la json
        split['train'].extend(traintemp)
        split['val'].extend(valtemp)
        split['test'].extend(testtemp)


        #salvare json cu split
        with open('split.json', 'w') as f:
            json.dump(split,f)
        




