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


# Use LabelEncoder for other categorical columns
label_encoder = LabelEncoder()
df['exercise habit'] = df['exercise habit'].map({
    'more than 3 months ago': 0,
    'less than 3 hours a week': 1,
    'no': 2,
    'never' : 3
})
df = pd.get_dummies(df, columns=['Frequency of alcohol consumption'], prefix='alcohol', drop_first=True)
df['Smoking habit'] = label_encoder.fit_transform(df['Smoking habit'])

for col in ['Age', 'kidney diseases', 'Fasting Blood sugar', 'Uri infection', 'profession', '#hours spent sitting per day','Smoking habit']:
    df[col] = label_encoder.fit_transform(df[col])

# Specify your features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the model
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Display text-based representation of the decision tree
tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

# Display graphical representation of the decision tree
dot_data = export_graphviz(clf, out_file=None, feature_names=list(X.columns), class_names=['normal', 'weak'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("fertility_tree6", format="png", cleanup=True)
graph.view("fertility_tree6")


