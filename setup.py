import os
import requests
import sys
import zipfile


def download_file(url, destination, file_name):
    if not os.path.exists(destination):
        os.makedirs(destination)

    destination = os.path.join(destination, file_name)

    print(f"Downloading {file_name} started")

    response = requests.get(url, stream=True)
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

    print(f"Downloading {file_name} completed")


def download_dataset():
    url = 'https://api.onedrive.com/v1.0/shares/u!aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBakxwRmctZjQ4bGpnYzFXU1dsMFhwWWZzWDBCTFE_ZT1vM3RWdmI/root/content'
    destination = 'data'
    file_name = 'DeepFashion.zip'

    download_file(url, destination, file_name)

    dataset_zip = destination + '/' + file_name
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(destination)

    os.remove(dataset_zip)


def download_pretrained_eva3d():
    url = 'https://drive.google.com/uc?id=1SYPjxnHz3XPRhTarx_Lw8SG_iz16QUMU'
    destination = 'checkpoint/512x256_deepfashion/volume_renderer'
    file_name = 'models_0420000.pt'

    download_file(url, destination, file_name)


def setup_eva3d_repo():
    sys.path.append('/content/2Dto3DAnimation/EVA3D')


if __name__ == "__main__":
    download_dataset()
    download_pretrained_eva3d()
    setup_eva3d_repo()
