import os
import shutil

def delete_annotation_folders(base_dir):
    # Walk through the base directory
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "annotations":
                # Construct the full path of the annotations folder
                annotations_path = os.path.join(root, dir_name)
                # Delete the annotations folder
                shutil.rmtree(annotations_path)
                print(f"Deleted {annotations_path}")

# Specify the base directory containing the split folders
base_directory = "/work/okupyn/VGGHeadNew/large"

# Call the function to delete annotation folders
delete_annotation_folders(base_directory)
