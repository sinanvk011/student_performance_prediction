import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

# Step 1: Create student dataset
data = {
    "Study_Hours": [
        1, 2, 3, 4, 5, 6, 7, 8, 2, 4,
        6, 7, 3, 5, 8, 1, 4, 6, 9, 10
    ],

    "Attendance": [
        50, 55, 60, 65, 70, 75, 80, 90, 60, 70,
        85, 90, 65, 75, 95, 45, 68, 82, 95, 98
    ],

    "Previous_Marks": [
        30, 35, 40, 45, 50, 60, 65, 80, 38, 48,
        62, 70, 42, 55, 85, 25, 46, 64, 88, 92
    ],

    "Final_Marks": [
        32, 38, 43, 48, 55, 65, 70, 85, 40, 50,
        68, 74, 45, 60, 88, 28, 49, 67, 91, 95
    ]
}

df = pd.DataFrame(data)

# Step 2: Select input and output
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Final_Marks"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create the model
model = LinearRegression()

# Step 5: Train the model
model.fit(X_train, y_train)

print("Model trained successfully!")

# Step 6: Test the model
predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", round(error, 2))
print("R2 Score:",round(r2_score(y_test,predictions),2))

# Step 7: Predict marks for a new student
new_student = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [85],
    "Previous_Marks": [70]
})

predicted_marks = model.predict(new_student)[0]

predicted_marks = max(0, min(100, predicted_marks))

print("Predicted Final Marks:", round(predicted_marks, 2))