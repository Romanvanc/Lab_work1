import os

def download_stanford_dogs():
    os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
    os.environ['KAGGLE_KEY'] = 'YOUR_API_KEY'
    
    os.system("kaggle datasets download -d jessicali9530/stanford-dogs-dataset")
    os.system("unzip -q stanford-dogs-dataset.zip -d stanford_dogs")
    print("Dataset downloaded to stanford_dogs/")

if __name__ == "__main__":
    download_stanford_dogs()
