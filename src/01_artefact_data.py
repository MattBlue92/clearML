


from pathlib import Path
import cv2
from clearml import Task, Dataset, TaskTypes
import matplotlib.pyplot as plt
import subprocess


task = Task.init(project_name="fruit_classification_with_vit", task_name="download_data", task_type= TaskTypes.data_processing)

output = subprocess.run(['echo', 'ciao'], capture_output=True, text=True)
print(output.stdout)
output = subprocess.run(['pwd'], capture_output=True, text=True)
print(output.stdout)
output = subprocess.run(['kaggle', 'datasets', '-d', 'shreyapmaher/fruits-dataset-images'], capture_output=True, text=True)
print(output.stdout)
output = subprocess.run(['cp', 'fruits-dataset-images.zip', '../data/fruits-dataset-images.zip'], capture_output=True, text=True)
print(output.stdout)
output = subprocess.run(['unzip', '../data/fruits-dataset-images.zip', '-d', '../data/new_fruit_dataset'], capture_output=True, text=True)
print(output.stdout)
output = subprocess.run(['rm', '../data/fruits-dataset-images.zip'], capture_output=True, text=True)
print(output.stdout)
output = subprocess.run(['rm', './fruits-dataset-images.zip'], capture_output=True, text=True)
print(output.stdout)

path_to_data = Path("..", "data", "new_fruit_dataset", "images")

folders = []
counts = []
for class_folder in path_to_data.iterdir():
    if class_folder.is_dir():
        name_folder = class_folder.name
        folders.append(name_folder)
        #path_to_save_class = path_to_save_data / name_folder
        #path_to_save_class.mkdir(exist_ok=True, parents=True)
        count = 0
        for image_path in class_folder.iterdir():
            if image_path.suffix != '.gif':
                #print(image_path.as_posix())
                count += 1
                #image = load_image(image_path.as_posix())
                #image_resized = resize_image(image, 224)
                #save_image(image_resized, (path_to_save_data / class_folder.name / image_path.name).as_posix())
        counts.append([count])

dataset = Dataset.create(dataset_name="fruit_dataset", dataset_project="fruit_classification_with_vit")
dataset.sync_folder(local_path=path_to_data.as_posix(), verbose=True)

dataset.get_logger().report_histogram(
    title="Number of images per class",
    series="Number of images",
    labels=folders,
    values=counts,
    xaxis="Class",
    yaxis="Count"
)


dataset.finalize(verbose=True, auto_upload=True)
task.close()
