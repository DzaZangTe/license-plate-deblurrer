import os
import shutil

def move_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for filename in os.listdir(src_dir):
        if filename[:-4].endswith('9'):
            src_path = os.path.join(src_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(src_path, dest_path)
            print(f'Moved: {src_path} -> {dest_path}')

src_blur_directory = 'training_blur'
dest_test_directory = 'testing_blur'
src_sharp_directory = 'training_sharp'
dest_sharp_directory = 'testing_sharp'

move_files(src_blur_directory, dest_test_directory)
move_files(src_sharp_directory, dest_sharp_directory)