from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import glob

data_folder = '/path/' 
path_to_byra = data_folder + 'byra_dataset/'
images = glob.glob(path_to_byra+'*.jpeg')

byra_df = pd.DataFrame(images,columns=['fullpath'])

filenames = []
for i in range(len(images)):
    names = images[i].split('/')
    filenames.append(names[8])
    
byra_df['filename'] = filenames

ids, classes, fat, image_no = [],[],[],[]
for j in range(len(filenames)):
    data = filenames[j].split('_')
    ids.append(int(data[1]))
    classes.append(int(data[3]))
    fat.append(int(data[5]))
    image_no.append(int(data[7].split('.')[0]))
    
byra_df['ID'], byra_df['Class'], byra_df['Fat'], byra_df['Img_no'] = ids, classes, fat, image_no 

row_indexes = byra_df[byra_df['Fat']>=66].index
byra_df.loc[row_indexes,'steatosis']= int(3)

row_indexes = byra_df[byra_df['Fat']<=65.9].index
byra_df.loc[row_indexes,'steatosis']= int(2)

row_indexes = byra_df[byra_df['Fat']<=33].index
byra_df.loc[row_indexes,'steatosis']= int(1)

row_indexes = byra_df[byra_df['Fat']<5].index
byra_df.loc[row_indexes,'steatosis']= int(0)

print('Distribution of Steatosis grades in the data:')
print(pd.value_counts(byra_df['steatosis'], normalize=True, dropna=False))

train_ids, val_ids = next(GroupShuffleSplit(n_splits=2, test_size=.15).split(byra_df,groups=byra_df['ID']))
train = byra_df.iloc[train_ids]
val = byra_df.iloc[val_ids]

train_list = ['train']*len(train)
val_list = ['val']*len(val)
train['Subset'] = train_list
val['Subset'] = val_list

splitset = pd.concat([train,val],axis=0)
splitset.to_csv(path_to_byra+'byra_dataset.csv')
