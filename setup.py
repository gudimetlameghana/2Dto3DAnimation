import os
import requests
import shutil


def move_eva3d_files():
    source_folder = 'EVA3D'
    destination_folder = 'eva3d'

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    file_names = list('eva3d_deepfashion.py', 'calculate_fid.py')

    for file_name in file_names:
        source_path = os.path.join(source_folder, file_name)
        destination_path = os.path.join(destination_folder, file_name)

        shutil.move(source_path, destination_path)

    shutil.rmtree(source_folder)


def download_pretrained_eva3d():
    url = 'https://drive.google.com/uc?id=1SYPjxnHz3XPRhTarx_Lw8SG_iz16QUMU'
    destination = 'models'
    file_name = 'models_0420000.pt'

    if not os.path.exists(destination):
        os.makedirs(destination)

    destination = os.path.join(destination, file_name)

    response = requests.get(url, stream=True)
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)


if __name__ == "__main__":
    move_eva3d_files()
    download_pretrained_eva3d()
