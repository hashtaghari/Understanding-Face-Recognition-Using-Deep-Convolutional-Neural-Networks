This folder contains the code and dataset we used to train the CNNs.

The models we trained were for:

1. CNN trained for face recognition, which has only seen inverted faces.
2. CNN trained for face recognition with upright images on the same dataset as above, to compare and contrast performances.
3. CNN trained for Face recognition, only trained on Asian faces.
4. CNN trained for Face recognition, only traiend on White faces.
5. CNN trained to recognise cars, i.e. classify a car based on its model.  
6. CNN trained for object categorisation, where face is one of the categories. 


### Datasets Used for training CNNs:

- CASIA-webface dataset was used for training a face recognition CNN, and we inverted all the images of the dataset to train a CNN for face recognition on inverted faces. 
  - Link to the dataset - [Click here.](https://www.kaggle.com/datasets/ntl0601/casia-webface)

- White faces are extracted from the CASIA dataset above by running a script based on the metadata provided. We used this dataset to train a CNN which has only seen white faces.
  
- Asian_face dataset was used for training the CNN on asian faces. 
  - Link to the dataset - [Click here.](https://www.kaggle.com/datasets/scienseenthusiast/asian-face)
  - We ran a script to put all identical faces in a folder in this dataset. (Every 5 images in order are images of the same person)

- To build a CNN that can recognise cars, we used the stanford cars dataset - [Click here.](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)


Here are the downloaded and cleaned datasets we used to trin the models: 

https://drive.google.com/drive/folders/1LiL78MVgy1vlQw5nkZZ1YqYNmZdzAj01?usp=share_link

https://drive.google.com/drive/folders/1csiJXa3iaOw5ImBplCBMdiyK2wglqggY?usp=share_link


