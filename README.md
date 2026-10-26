[README_ML_project.md](https://github.com/user-attachments/files/33962087/README_ML_project.md)

# Machine Learning Project

This project demonstrates a basic machine learning workflow using the **Iris dataset**. It includes data analysis, visualization, preprocessing, classification, a small regression example, model evaluation, hyperparameter tuning, and a Streamlit app for interactive predictions.

## What This Project Includes

- Exploratory Data Analysis (EDA)
- Missing-value and duplicate checks
- Histograms, boxplots, correlation heatmap, and outlier analysis
- Train-test split with stratification
- Min-Max scaling and Standardization
- 5-fold cross-validation
- L1 and L2 regularization experiments
- Classification using Logistic Regression, Random Forest, and SVM
- Classification evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrices
- Hyperparameter tuning with `GridSearchCV`
- A Linear Regression example to predict sepal length
- Saving models and schemas for a Streamlit prediction interface

## Dataset

The notebook uses the built-in **Iris dataset** from `scikit-learn`, so a separate CSV file is not required.

The classification task predicts one of three Iris flower classes:

- Setosa
- Versicolor
- Virginica

The regression example predicts `sepal length (cm)` using the other Iris measurements as input features.

## Libraries Used

- **NumPy** — numerical calculations
- **Pandas** — data handling and analysis
- **Matplotlib** — charts and plots
- **Seaborn** — statistical visualizations
- **Scikit-learn** — preprocessing, machine learning models, metrics, cross-validation, and tuning
- **Joblib** — saving and loading trained models
- **Streamlit** — interactive web app

The notebook also uses Python standard-library modules such as `json`, `warnings`, and `os`. These do not need separate installation.

## Installation

Install the required packages:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit notebook
```

## How to Run

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `ML_project.ipynb` and run the cells in order.
5. After the notebook has generated the model files and `streamlit_app.py`, run the app:

   ```bash
   streamlit run streamlit_app.py
   ```

## Model Artifacts

The notebook is set up to save classification and regression models, scalers, and JSON schemas in the `MODEL_FILES/` folder. Generate these files by running the relevant notebook cells before starting the Streamlit app.

## Note

The notebook's Streamlit page title says “Dynamic Placement & Salary Predictor,” but the data currently used in the notebook is the Iris dataset and the regression target is sepal length—not student placement or salary. Update the app title and prediction wording if you intend to present this as an Iris classifier.
