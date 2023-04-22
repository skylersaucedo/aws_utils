"""
utils harvested from Sagemaker notebook.
"""

## cycle through each csv to generate updated df for lst file
# index header_column label_width className xmin ymin xmax ymax path 
import random
import csv
import pandas as pd
import os

def convert_csv_to_lst(df, train_or_test, lstname):
    """
    Use original df from LabelImg and xml to csv and reformat to .lst standards
    https://cv.gluon.ai/build/examples_datasets/detection_custom.html

    """

    df_lst = []

    for idx, row in df.iterrows():

        header_column = 2
        label_width = 5
        pth = root + '/' + train_or_test + '/' + row['filename']
        w = row['width']
        h = row['height']

        xmin = row['xmin']
        xmax = row['xmax']
        ymin = row['ymin']
        ymax = row['ymax']

        n_xmin = xmin / w
        n_xmax = xmax / w
        n_ymin = ymin / h
        n_ymax = ymax / h

        width = n_xmax - n_xmin
        height = n_ymax - n_ymin

        className = row['class']

        df_lst.append([header_column,label_width,className,n_xmin,n_ymin,n_xmax,n_ymax,pth])
        
    raw_df = pd.DataFrame(data=df_lst, columns=["header_cols","label_width","className","XMin","YMin","XMax","YMax","ImagePath"],dtype = object)

    # now we need to reformat into gluon, each record is an image with lots of objects 
    
    unique_list = raw_df.ImagePath.unique()
    
    final = []
    for img_path in unique_list:

        df_rows = raw_df.loc[raw_df['ImagePath'] == img_path]
        the_img_path = df_rows.loc[df_rows.index[0]]['ImagePath']
        r = random.randint(0,10000000)
        length = len(df_rows)
        count = 1
        arr = [r,2,5]

        for index, row in df_rows.iterrows():
            xmin = str(row['XMin'])
            ymin = str(row['YMin'])
            xmax = str(row['XMax'])
            ymax = str(row['YMax'])
            className = str(row['className'])

            arr.extend([className, xmin, ymin, xmax, ymax, the_img_path])

            if count == length:
                arr.append(the_img_path)
            count+=1

        final.append(arr)
    # now write out .lst file
    
    with open(lstname, 'w', newline = '') as out:
        for row in final:
            writer = csv.writer(out, delimiter = '\t')
            writer.writerow(row)
            
    print('.lst is made!')
    

#df_train_unformatted = pd.read_csv(root + '/' + 'train_labels.csv')
#df_test_unformatted = pd.read_csv(root + '/' + 'test_labels.csv')

#convert_csv_to_lst(df_train_unformatted, 'train', 'train.lst')
#convert_csv_to_lst(df_test_unformatted, 'test', 'test.lst')