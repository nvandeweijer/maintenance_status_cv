# Predict the property's maintenance status using images

In this project, the maintenance status of a property will be predicted using images. For this purpose, the dataset in csv format of Clappform is used.

## Download images
To download photos, the script `download_images.py` can be used. This downloads the photos and then classifies them by maintenance status into folders.

## Create JSON file
The splits must be created before training begins; this will be done by house. `make_clean_data.py` will be used to carry it out. The file format will be JSON.

### Room Classification
When creating the JSON file, the room type predictions are added for every image. Another approach is that some room types can be excluded. [This code](https://github.com/tae898/room-classification) and a trained model, which is also available for download there, are used for this. 
The images below a specific percentage were moved to another folder using `roomtype/exclude_roomtypes.py` and `roomtype/roomtype_classifier.py`.

## Training
The training is done with and without the package [Cleanlab](https://cleanlab.ai/). The script with Cleanlab is called `train_with_cleanlab.py`.
The script for training without Cleanlab using [Pytorch Lightning](https://www.pytorchlightning.ai/) is called `train_without_cleanlab.py`.
