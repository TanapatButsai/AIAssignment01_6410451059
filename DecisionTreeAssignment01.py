import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import graphviz

# Load your dataset (replace 'your_dataset.csv' with the actual filename)
df = pd.read_csv('fertility2.csv')

# Display the first few rows of your dataset
print(df.head())

# Use LabelEncoder for each categorical column
label_encoder = LabelEncoder()
for col in ['Age', 'kidney diseases', 'Fasting Blood sugar', 'Uri infection', 'exercise habit', 'Frequency of alcohol consumption', 'Smoking habit', 'profession', '#hours spent sitting per day']:
    df[col] = label_encoder.fit_transform(df[col])

# Display the first few rows of your dataset
print(df.shape)


