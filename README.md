
# Insurance Fraud Detection

This project aims to detect fraudulent insurance claims using machine learning techniques. The repository contains the code for data analysis, preprocessing, model training, and a web application for interacting with the trained model.

## Overview

The primary objective of this project is to develop a model to detect fraudulent insurance claims. This involves using machine learning techniques to analyze historical data and identify patterns that indicate fraud.

## Components

### Data Analysis and Preprocessing

- **Exploratory Data Analysis (EDA)**: Understanding the data distribution, detecting anomalies, and visualizing relationships between different variables.
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
- **Feature Selection**: Identifying important features using techniques like Extra Trees Regressor.

### Model Training

- **Model Training**: Splitting the data into training and testing sets, and training machine learning models.
- **Model Evaluation**: Evaluating the performance of the trained models using metrics like accuracy, classification report, and confusion matrix.

### Web Application

- **Flask Web App**: A web interface for uploading data, making predictions, and visualizing results.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- Necessary Python libraries:
  - Pandas
  - Matplotlib
  - Seaborn
  - Scikit-learn
  - Flask
  - TensorFlow (if used)
  - Flask-Material

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/insurance-fraud-detection.git
   cd insurance-fraud-detection
   ```

2. **Install required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place the dataset:**

   Ensure the `insurance_claims.csv` file is in the `data` directory.

### Running the Jupyter Notebook

1. **Navigate to the notebook directory:**

   ```bash
   cd notebooks
   ```

2. **Open the Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

3. **Run the `Insurance Fraud Detection.ipynb` notebook:**

   Execute the cells sequentially to perform data analysis, preprocessing, model training, and evaluation.

### Running the Web Application

1. **Ensure the necessary templates are in the `templates` directory:**
   - `index.html`
   - `about.html`
   - `upload.html`
   - `uploaded.html`

2. **Run the Flask application:**

   ```bash
   python main.py
   ```

3. **Access the web application:**

   Open your web browser and go to `http://127.0.0.1:5000/`.

## File Structure

```
insurance-fraud-detection/
├── data/
│   └── insurance_claims.csv
├── notebooks/
│   └── Insurance Fraud Detection.ipynb
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── upload.html
│   └── uploaded.html
├── main.py
├── requirements.txt
└── README.md
```


### Code Explanation

#### Imports and Setup
The script imports various libraries, including Flask for web development, Scikit-learn for machine learning, and other utilities like Pandas for data manipulation.

```python
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = '1a2b3c4d5e'
```

#### Routes
The script defines several routes to handle different parts of the web application:

- **Home Route**:
  ```python
  @app.route('/')
  def home():
      return render_template('index.html')
  ```

- **About Route**:
  ```python
  @app.route('/about')
  def about():
      return render_template('about.html')
  ```

#### File Upload Handling
The script likely includes functionality for users to upload insurance claims data for analysis. This part of the code will handle file uploads and data processing:

```python
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join('uploads', filename))
            # Process the uploaded file here
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('upload.html')
```

#### Model Prediction
The script includes logic to load the pre-trained model and make predictions on new data:

```python
@app.route('/predict', methods=['POST'])
def predict():
    # Load data from the request
    data = request.get_json()
    # Process and predict using the loaded model
    prediction = model.predict([data])
    return jsonify({'prediction': prediction.tolist()})
```

### Full Code Structure
The overall structure of the `main.py` script seems to involve:
1. Setting up the Flask application.
2. Defining routes for the home page, about page, file upload, and prediction.
3. Handling file uploads and saving them to a specific directory.
4. Loading the trained machine learning model and making predictions based on user inputs.
5. Rendering HTML templates to display the results and provide an interface for user interaction.

### Running the Script
To run the Flask application, execute the script using Python:

```bash
python main.py
```

Ensure that the necessary templates (`index.html`, `about.html`, `upload.html`) are present in the `templates` directory and the static files (CSS, JS) are in the `static` directory.

Would you like a more detailed breakdown of any specific part of the code or further assistance with anything else?
## Conclusion
 
 This documentation provides a comprehensive guide to setting up and running the Insurance fraud Prediction System. By following the steps outlined, you should be able to deploy the application and make predictions based on user input. If you encounter any issues, ensure that all dependencies are installed and that the model file is correctly placed in the `models` directory.



## Result-Screenshots

![data_science](https://github.com/user-attachments/assets/c893535f-53e8-46e6-80a8-c35cfb5f849e)
![data science](https://github.com/user-attachments/assets/3f458714-65b3-49c5-bebc-14848b08683a)
![data-science](https://github.com/user-attachments/assets/48f48324-5448-46a6-b99b-c3fd1c05e7cb)

## Supervised by 
[Prof. Agughasi Victor Ikechukwu](https://github.com/Victor-Ikechukwu), 
(Assistant Professor) 
Department of CSE, MIT Mysore)


## Collaborators

- 4MH21CS044 [Likith Nirvan]() 
- 4MH21CS126 [Guru Prasad G M](https://github.com/Guruprasad619)
- 4MH21CS008 [Arun Ram R](https://github.com/ArunramR)
- 4MH21CS037 [Jayanthan B N](https://github.com/Jayanthan23)

## License

This project is licensed under the MIT License.

---
