import os
import requests


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
    download_pretrained_eva3d()
