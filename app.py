from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained machine learning model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":
        try:
            study_hours = float(request.form["study_hours"])
            attendance = float(request.form["attendance"])
            previous_marks = float(request.form["previous_marks"])

            if not 0 <= study_hours <= 24:
                raise ValueError("Study hours must be between 0 and 24.")

            if not 0 <= attendance <= 100:
                raise ValueError("Attendance must be between 0 and 100.")

            if not 0 <= previous_marks <= 100:
                raise ValueError("Previous marks must be between 0 and 100.")

            student = pd.DataFrame({
                "Study_Hours": [study_hours],
                "Attendance": [attendance],
                "Previous_Marks": [previous_marks]
            })

            result = model.predict(student)[0]

            prediction = round(max(0, min(100, result)), 2)

        except (ValueError, TypeError):
            error = "Please enter valid numbers within the allowed ranges."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)