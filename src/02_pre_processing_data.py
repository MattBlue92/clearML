from pathlib import Path
from clearml import Task, Dataset, TaskTypes
import pandas as pd
import numpy as np
import shutil


task = Task.init(project_name="fruit_classification_with_vit", task_name="pre_process_data", task_type= TaskTypes.data_processing)

##download data from clearML
dataset = Dataset.get(dataset_id='cfa7b0b9054e4d6baed1260e12eb7cbd')


dataset.get_mutable_local_copy('../data/fruit_dataset')


path_to_data = Path('..',"data", 'fruit_dataset')
choices = ['train_set', 'val_set', 'test_set']
probabilities = [0.7, 0.15, 0.15] # possono diventare dei paramatri da selezionare con argparse
seed = 42
np.random.seed(42) #posso impostare argparser

resources = []
folders = []
filenames = []
sets = []

for folder in path_to_data.iterdir():
    if folder.is_dir():
        # folders.append(folder.name)
        for resource in folder.iterdir():
            if resource.is_file():
                resources.append(resource.as_posix())
                filenames.append(resource.name)
                folders.append(folder.name)
        # num_images = len(list(folder.glob("*.jpg")))
        # result = np.random.choice(choices, size=num_images, p=probabilities)
        train_set = ['train_set'] * 28
        val_set = ['val_set'] * 6
        test_set = ['test_set'] * 6
        result = train_set + val_set + test_set
        result = np.random.permutation(result)
        sets.extend(result)

data = [resources, filenames, folders, sets]
df = pd.DataFrame(data, index=["resources", "filenames", "folders", "sets"]).T
df


def new_filename(row):
    filename = row['filenames']
    name, ext = filename.split(".")
    new_filename = f"{name}-{row['folders']}.{ext}".lower()
    return new_filename


df['folders'] = df['folders'].str.replace(" ", "_")
df['new_filenames'] = df.apply(new_filename, axis=1)
df



path_to_store_data = Path('..', "data", "fruit_dataset_preprocessed")
path_to_store_data.mkdir(exist_ok=True, parents=True)
path_to_store_data.joinpath("train_set").mkdir(exist_ok=True)
path_to_store_data.joinpath("val_set").mkdir(exist_ok=True)
path_to_store_data.joinpath("test_set").mkdir(exist_ok=True)
for idx, row in df.iterrows():
    old_path = row['resources']
    new_path = path_to_store_data.joinpath(row['sets'], row['new_filenames'])
    shutil.copy(old_path, new_path)



training_set = Dataset.create(dataset_name="fruit_dataset", dataset_project="fruit_classification_with_vit", parent_datasets=[dataset.id], dataset_tags= ['training_set'])
validation_set = Dataset.create(dataset_name="fruit_dataset", dataset_project="fruit_classification_with_vit", parent_datasets=[dataset.id], dataset_tags= ['validation_set'])
test_set = Dataset.create(dataset_name="fruit_dataset", dataset_project="fruit_classification_with_vit", parent_datasets=[dataset.id],
                          dataset_tags= ['test_set'])


training_set.sync_folder(local_path=path_to_store_data.joinpath('train_set').as_posix(), verbose=True)
validation_set.sync_folder(local_path=path_to_store_data.joinpath('val_set').as_posix(), verbose=True)
test_set.sync_folder(local_path=path_to_store_data.joinpath('test_set').as_posix(), verbose=True)


training_set.finalize(verbose=True, auto_upload=True)

validation_set.finalize(verbose=True, auto_upload=True)

test_set.finalize(verbose=True, auto_upload=True)


task.close()