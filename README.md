
# MOVIE RATING PREDICTION WITH PYTHON - CODSOFT INTERNSHIP TASK

This project aims to predict movie ratings based on various features such as year, duration, votes, genre, director, and main actors. The prediction model is built using a Random Forest Regressor and is deployed using Streamlit for an interactive user experience.


## Description


The project consists of two main files:

 1) model.py: This file contains the code to train and run the machine learning model efficiently.

 2) app.py: This file sets up a Streamlit web application for an interactive and user-friendly interface to input movie details and predict ratings.
The dataset used for this project is IMDb Movies India.csv, which contains various features related to Indian movies.

You can run either model.py for a command-line interface or app.py for a graphical user interface. app.py provides a better user experience by using Streamlit to interact with the model.
## Acknowledgements

We would like to thank the following resources and individuals for their contributions and support:

1) IMDb: For providing the data used in this project.

2) Streamlit: For offering an easy-to-use framework for deploying machine learning models.

3) Scikit-learn: For providing powerful machine learning tools and libraries.


## Demo

Link to demo : https://photos.app.goo.gl/hn2ihjiBHKAPrsES8

You can see a live demo of the application by running the app.py file. The Streamlit app allows you to input movie details and get a predicted rating based on the trained model.
## Features

1) Data Loading and Preprocessing: The model can load and preprocess data from the IMDb Movies India.csv file.

2) Model Training: Utilizes a Random Forest Regressor to train the model on movie data.

3) Interactive User Input: Through the Streamlit app, users can input movie details and receive a predicted rating.

4) Feature Importance: Displays the importance of each feature in predicting movie ratings.


## Technologies Used

Python: The programming language used to implement the model and the Streamlit app.

Pandas: For data manipulation and preprocessing.

NumPy: For numerical operations.

Scikit-learn: For building and training the machine learning 
model.

Streamlit: For creating the interactive web application.

Random Forest Regressor: The machine learning algorithm used for predicting movie ratings.
## Installation

To get started with this project, follow these steps:

1) Clone the repository:

git clone https://github.com/SiddharthWayne/MOVIE-RATING-PREDICTION-WITH-PYTHON---CODSOFT.git

cd movie-rating-prediction

2) Install the required packages:

pip install -r requirements.txt

Ensure that requirements.txt includes the necessary dependencies like pandas, numpy, scikit-learn, and streamlit.

3) Download the dataset:

Place the IMDb Movies India.csv file in the project directory. Make sure the path in model.py and app.py is correctly set to this file.


## Usage/Examples

Running the Model (model.py):

To train and run the model using the command line, execute the following:
python model.py

This will train the model and allow you to input movie details via the command line interface to get a predicted rating.

Running the Streamlit App (app.py):

To run the Streamlit app for an interactive experience, execute the following:streamlit run app.py

This will start the Streamlit server, and you can open your web browser to the provided local URL to use the app.


Example:

Once the Streamlit app is running, you can input details such as:

Year: 2020

Duration: 120 minutes

Votes: 10000

Genre: Action

Director: Christopher Nolan

Actor 1: Leonardo DiCaprio

Actor 2: Joseph Gordon-Levitt

Actor 3: Ellen Page

Click the "Predict Rating" button to get the predicted rating for the movie.