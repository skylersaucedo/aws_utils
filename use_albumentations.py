"""
Use this to create various permutations of images with bounding boxes

https://www.kaggle.com/code/ankursingh12/data-augmentation-for-object-detection

"""

import numpy as np
import imageio
import os
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import pandas as pd

#paths

df_test_labels_path = r'C:\Users\endle\Desktop\tsi_thread_defects\models\research\object_detection\images\test_labels.csv'
df_train_labels_path = r'C:\Users\endle\Desktop\tsi_thread_defects\models\research\object_detection\images\train_labels.csv'
train_dir = r'C:\Users\endle\Desktop\tsi_thread_defects\models\research\object_detection\images\train'
test_dir = r'C:\Users\endle\Desktop\tsi_thread_defects\models\research\object_detection\images\test'

alb_dir = r'C:\Users\endle\Desktop\tsi_thread_defects\models\research\object_detection\albumentations_test'
imgs_path = r'C:\Users\endle\Desktop\tsi_thread_defects\models\research\object_detection\images'

def load_data(csv_path):
    train_df = pd.read_csv(csv_path)
    df = train_df.rename({'filename' : 'image_id', 'class' : 'source'}, axis=1)

    return df

def draw_rect(img, bboxes, color=(255, 0, 0)):
    img = img.copy()
    for bbox in bboxes:
        bbox = np.array(bbox).astype(int)
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img

def read_img(img_id):
    img_path = train_dir +str(img_id)+'.jpg'
    img = cv2.imread(img_path)
    return img

def read_bboxes(df, img_id):
    return df.loc[df.image_id == img_id+'.jpg', 'xmin ymin xmax ymax'.split()].values

def read_width_height_class(df, img_id):

    x = df.loc[df.image_id == img_id+'.jpg', 'width height source'.split()].values
    return x[0][0], x[0][1], x[0][2]
    
def plot_img(img_id, bbox=False):
    img= read_img(img_id)
    if bbox:
        bboxes = read_bboxes(img_id)
        img    = draw_rect(img, bboxes)
    plt.imshow(img)
    
def plot_multiple_img(img_matrix_list, title_list, ncols, nrows=3, main_title="img_id"):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        
        myaxes[i // ncols][i % ncols].imshow(img)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
        myaxes[i // ncols][i % ncols].grid(False)
        myaxes[i // ncols][i % ncols].set_xticks([])
        myaxes[i // ncols][i % ncols].set_yticks([])

    savepath = alb_dir + "//" +  main_title + ".jpg"

    print(savepath)

    plt.savefig(savepath)
    plt.show()


def main():

    csv_paths = [df_train_labels_path, df_test_labels_path]
    dir_paths = [train_dir,test_dir]
    albs = ['train_alb', 'test_alb']

    for k, directory in enumerate(dir_paths):
        
        df = load_data(csv_paths[k])

        album_anno_list = []
        
        x = [r.split('.')[0] for r in os.listdir(dir_paths[k])]

        for img_id in x:
            
            img_path = dir_paths[k] + '\\' + str(img_id) + '.jpg'
            chosen_img = cv2.imread(str(img_path))
            bboxes = read_bboxes(df, img_id)
            w, h, c = read_width_height_class(df, img_id)
            bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}

            albumentation_list = [A.Compose([A.RandomFog(p=1)], bbox_params=bbox_params),
                                A.Compose([A.RandomBrightnessContrast(p=1)], bbox_params=bbox_params),
                                A.Compose([A.RandomCrop(p=1, height=512, width=512)], bbox_params=bbox_params), 
                                A.Compose([A.Rotate(p=1, limit=90)], bbox_params=bbox_params),
                                A.Compose([A.RGBShift(p=1)], bbox_params=bbox_params), 
                                A.Compose([A.RandomSnow(p=1)], bbox_params=bbox_params),
                                A.Compose([A.VerticalFlip(p=1)], bbox_params=bbox_params), 
                                A.Compose([A.RandomContrast(limit=0.5, p = 1)], bbox_params=bbox_params)
                                ]

            titles_list = ["Original", 
                        "RandomFog",
                        "RandomBrightness", 
                        "RandomCrop",
                        "Rotate", 
                        "RGBShift", 
                        "RandomSnow", 
                        "VerticalFlip", 
                        "RandomContrast"]

            img_matrix_list = [draw_rect(chosen_img, bboxes)]

            for i, aug_type in enumerate(albumentation_list):
                anno = aug_type(image=chosen_img, bboxes=bboxes, labels=np.ones(len(bboxes)))

                #print(anno['bboxes'])
                f_name = str(img_id) +"_"+ titles_list[i] +  ".jpg"
                savepath = alb_dir + "//" +  f_name

                for j, val in enumerate(anno['bboxes']):
                    b = anno['bboxes'][j]

                    # write to list

                    x_min = int(b[0])
                    y_min = int(b[1])
                    x_max = int(b[2])
                    y_max = int(b[3])

                    album_anno_list.append([f_name, w, h, c, x_min, y_min, x_max, y_max])

                    

                #img  = draw_rect(anno['image'], anno['bboxes'])

                # write to file

                imageio.imwrite(savepath, anno['image'])

                #plt.imshow(anno['image'])
                #plt.show()

                

                #print(savepath)
                #plt.savefig(savepath)

                #img_matrix_list.append(img)

            # plot_multiple_img(img_matrix_list, 
            #                 titles_list, 
            #                 ncols = 3, 
            #                 main_title=img_id)

        df_alb = pd.DataFrame(album_anno_list, columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        
        csv_path = imgs_path + '\\' + albs[k] + '_.csv'
        df_alb.to_csv(csv_path)


if __name__ == '__main__':
    main()