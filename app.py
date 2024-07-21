import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# Load and preprocess the data
@st.cache_data
def load_data():
    data_path = r"C:\Users\siddh\OneDrive\Desktop\codsoft intern\MOVIE RATING PREDICTION WITH PYTHON\IMDb Movies India.csv"
    
    # Try different encodings
    encodings = ['utf-8', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            data = pd.read_csv(data_path, encoding=encoding)
            st.success(f"Successfully loaded the data using {encoding} encoding.")
            break
        except UnicodeDecodeError:
            continue
    else:
        st.error("Failed to load the data with any of the attempted encodings.")
        return None
    
    # Clean the 'Year' column
    data['Year'] = data['Year'].str.extract('(\d+)').astype(float)
    
    # Clean the 'Duration' column (assuming it's in the format '123 min')
    data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)
    
    # Convert 'Votes' to numeric, replacing any non-numeric values with NaN
    data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
    
    return data



# Train the model
@st.cache_resource
def train_model(data):
    features = ['Year', 'Duration', 'Votes', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
    target = 'Rating'
    
    # Handle missing values
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

# Streamlit app
def main():
    st.title("Movie Rating Prediction")
    
    data = load_data()
    model, features = train_model(data)
    
    st.sidebar.header("Input Movie Details")
    
    # Input fields
    year = st.sidebar.number_input("Year", min_value=1900, max_value=2023, value=2020)
    duration = st.sidebar.number_input("Duration (minutes)", min_value=1, max_value=300, value=120)
    votes = st.sidebar.number_input("Number of Votes", min_value=1, max_value=1000000, value=10000)
    genre = st.sidebar.selectbox("Genre", data['Genre'].unique())
    director = st.sidebar.selectbox("Director", data['Director'].unique())
    actor1 = st.sidebar.selectbox("Actor 1", data['Actor 1'].unique())
    actor2 = st.sidebar.selectbox("Actor 2", data['Actor 2'].unique())
    actor3 = st.sidebar.selectbox("Actor 3", data['Actor 3'].unique())
    
    # Create a dataframe with user input
    input_data = pd.DataFrame({
        'Year': [year],
        'Duration': [duration],
        'Votes': [votes],
        'Genre': [genre],
        'Director': [director],
        'Actor 1': [actor1],
        'Actor 2': [actor2],
        'Actor 3': [actor3]
    })
    
    # Make prediction
    if st.sidebar.button("Predict Rating"):
        prediction = model.predict(input_data)
        
        st.subheader("Prediction Result")
        st.write(f"The predicted rating for this movie is: {prediction[0]:.2f}")
        
        # Display input data
        st.subheader("Input Movie Details")
        st.write(input_data)
        
        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = model.named_steps['regressor'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))

if __name__ == "__main__":
    main()