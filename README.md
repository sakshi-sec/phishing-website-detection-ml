# ML-Based Phishing Website Detection System

A machine learning project that classifies a website as **phishing** or **legitimate**
from 30 website characteristics, with a Streamlit web app that shows the prediction,
probability, risk level and key phishing indicators.

## Results
| Item | Value |
|---|---|
| Best model | Random Forest |
| Test accuracy | 94.02% |
| 5-fold cross-validation accuracy | 94.89% ± 0.39% |
| Models compared | Logistic Regression, Decision Tree, KNN, Naive Bayes, SVM, Random Forest |

5,206 duplicate rows were removed before training (11,055 -> 5,849 unique records)
to avoid train/test leakage. The split is 80:20 and stratified.

## Project files
- `train_model.py` - cleans the data, trains and compares 6 models, saves the best one
- `app.py` - Streamlit web application
- `phisingData.csv` - dataset
- `requirements.txt` - Python libraries

## How to run
```
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset
Phishing Websites dataset, UCI Machine Learning Repository
(R. Mohammad, F. Thabtah, L. McCluskey).

## Limitations
The feature values are entered by the user; the app does not extract them from a URL.
Results are decision support, not a guarantee that a website is safe.

## Author
Sakshi Wagh
