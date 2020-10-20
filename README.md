# MusicGenreClassifier

We did data scraping from YouTube playlists to collect dataset with songs from 9 genres including: classical, hip hop/rap, metal, pop, jazz, techno, rock, turbo  folk and Macedonian folklore songs. Then we preprocessed the data, made feature extraction and classification with different methods. All this can be found in the following notebooks:

1. data_scrape.ipynb: obtaining the songs from all the genres via scraping algorithms in Python.
2. data_preprocess.ipynb: plots, transformations and data cleansing.
3. k_means_genres.ipynb: unsupervised approach for clustering the data in 9 clusters using the K-Means algorithm and T-SNE
4. Music_genre_classification_ML.ipynb: classical machine learning techniques for classification
5. Genre_classifier_1.ipynb: deep learning CNN model for classification.


We also created application for automatic classification of given song to a particular genre. It was developed with Fast API, the user sends post request with uploading a song to the website, the server will return a result with probability about the genre of the song. To activate this web app you need to write "uvicorn main:app --reload" in the terminal
