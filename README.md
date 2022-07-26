# Predict the property's maintenance status using images

In this project, the maintenance status of a property will be predicted using images. For this purpose, the dataset in csv format of Clappform is used.



## Download images
To download photos, the script `download_images.py` can be used. This downloads the photos and then classifies them by maintenance status into folders.

## Room Classification
Some room types can be excluded to improve the classifier. [This code](https://github.com/tae898/room-classification) and a trained model, which is also available for download there, are used for this. 
The images below a specific percentage were moved to another folder using `roomtype/exclude_roomtypes.py` and `roomtype/roomtype_classifier.py`.


## Create JSON files
The splits must be created before training begins; this will be done by house. `make_clean_data.py` will be used to carry it out. The file format will be JSON.

## Training
