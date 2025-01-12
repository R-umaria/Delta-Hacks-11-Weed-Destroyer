import os
import shutil

# Define the paths
source_folder = '/Users/sam/code/weedDetection/agri_data/data'
weed_folder = '/Users/sam/code/weedDetection/weed'
crop_folder = '/Users/sam/code/weedDetection/crop'

# Create destination folders if they don't exist
os.makedirs(weed_folder, exist_ok=True)
os.makedirs(crop_folder, exist_ok=True)

# Iterate over files in the source folder
for filename in os.listdir(source_folder):
    print(filename)
    if filename.endswith('.jpeg'):  # Assuming image files are .jpg
        # Corresponding text file
        text_file = os.path.splitext(filename)[0] + '.txt'
        text_file_path = os.path.join(source_folder, text_file)

        print(text_file + " exists: " + str(os.path.exists(text_file_path)))
        
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r') as file:
                first_char = file.read(1)
                
                if first_char == '0':
                    shutil.move(os.path.join(source_folder, filename), crop_folder)
                    shutil.move(text_file_path, crop_folder)
                    print("Moved to crop")
                elif first_char == '1':
                    shutil.move(os.path.join(source_folder, filename), weed_folder)
                    shutil.move(text_file_path, weed_folder)
                    print("Moved to weed")