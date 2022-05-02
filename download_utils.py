import gdown
import os
import zipfile
import tarfile

def download_dataset():
    #id = "1B-2EUMIYy4vJCncmBKL-G6B6Keq4jLXT"
    #file = "drinks.zip"
    id = "1AdMbVK110IKLG7wJKhga2N2fitV1bVPA"
    url = f'https://drive.google.com/uc?id={id}'
    file = "drinks.tar.gz"

    if os.path.exists("drinks"):
        if(os.path.exists(file)): os.remove(file)
        print("Dataset already downloaded.")
        return

    if not os.path.exists(file):
        gdown.download(url, file, quiet=False)

    _extract_targz(file)
    
def _extract_targz(file_path):
    with open(file_path, 'rb', ) as f:
        print("Extracting drinks.tar.gz ...")
        file = tarfile.open(fileobj=f, mode="r|gz")
        file.extractall(path="")
        print("Extraction done.")
    os.remove(file_path)

def _extract_zip(file):
    with zipfile.ZipFile(file, 'r') as f:
        print("Extracting drinks.tar.gz ...")
        f.extractall(path="")
        print("Extraction done.")
    os.remove(file)


def download_pretrained_model():
    
    id = "1mRtbvLYKAl1dMQl7BORIyYIitezD5WPw"
    url = f'https://drive.google.com/uc?id={id}'
    file = "checkpoints/pretrained.pth"
    
    if not os.path.exists(file):
        if not os.path.exists("checkpoints"): os.mkdir("checkpoints")
        gdown.download(url, file, quiet=False)
    else:
        print("Pretrained.pth file already downloaded.")
