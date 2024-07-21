import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def load_data():
    data_path = r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\MOVIE RATING PREDICTION WITH PYTHON\IMDb Movies India.csv"
    
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(data_path, encoding=encoding)
            print(f"Successfully loaded the data using {encoding} encoding.")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("Failed to load the data with any of the attempted encodings.")
        return None
    
    data['Year'] = data['Year'].str.extract('(\d+)').astype(float)
    data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)
    data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
    
    return data

def train_model(data):
    features = ['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    target = 'Rating'
    
    data = data.dropna(subset=features + [target])
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['Year', 'Duration', 'Votes']
    categorical_features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    return model, X.columns

def get_user_input(data):
    print("\nEnter Movie Details:")
    year = int(input("Year (1900-2023): "))
    duration = int(input("Duration (minutes, 1-300): "))
    votes = int(input("Number of Votes (1-1000000): "))
    genre = input("Genre: ")
    director = input("Director: ")
    actor1 = input("Actor 1: ")
    actor2 = input("Actor 2: ")
    actor3 = input("Actor 3: ")
    
    return pd.DataFrame({
        'Year': [year],
        'Duration': [duration],
        'Votes': [votes],
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    })

def main():
    print("Movie Rating Prediction")
    
    data = load_data()
    if data is None:
        return
    
    model, features = train_model(data)
    
    while True:
        input_data = get_user_input(data)
        
        prediction = model.predict(input_data)
        
        print(f"\nPrediction Result:")
        print(f"The predicted rating for this movie is: {prediction[0]:.2f}")
        
        print("\nInput Movie Details:")
        print(input_data)
        
        print("\nFeature Importance:")
        feature_importance = model.named_steps['regressor'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        print(importance_df)
        
        another = input("\nDo you want to predict another movie? (yes/no): ").lower()
        if another != 'yes':
            break

if __name__ == "__main__":
    main()