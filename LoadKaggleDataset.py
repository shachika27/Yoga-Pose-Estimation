'''
Python code to load Kaggle Dataset containing 5 Yoga Pose classes

Total training images-
	Downdog/  223 images
	Goddess/ 180 images
	Plank/ 266 images
	Tree/ 160 images
	Warrior/ 252 images

Total test images-
	Downdog/  97 images
	Goddess/ 80 images
	Plank/ 115 images
	Tree/ 69 images
	Warrior/ 109 images
'''
from google.colab import drive
drive.mount('/content/gdrive')

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Colab Notebooks/Kaggle"

%cd /content/gdrive/My Drive/Colab Notebooks/Kaggle

!kaggle datasets download -d niharika41298/yoga-poses-dataset

!unzip \*.zip  && rm *.zip