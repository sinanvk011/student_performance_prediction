import pandas as pd
import pickle

from sklearn.linear_model import LinearRegression

# Create the student dataset
data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
                    6, 7, 3, 5, 8, 1, 4, 6, 9, 10],

    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 60, 70,
                   85, 90, 65, 75, 95, 45, 68, 82, 95, 98],

    "Previous_Marks": [30, 35, 40, 45, 50, 60, 65, 80, 38, 48,
                       62, 70, 42, 55, 85, 25, 46, 64, 88, 92],

    "Final_Marks": [32, 38, 43, 48, 55, 65, 70, 85, 40, 50,
                    68, 74, 45, 60, 88, 28, 49, 67, 91, 95]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Select input and output
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
print("model.pkl created successfully!")