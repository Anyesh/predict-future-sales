# Predict Future Sales

This challenge serves as final project for the "How to win a data science competition" Coursera course.

In this competition you will work with a challenging time-series dataset consisting of daily sales data, kindly provided by one of the largest Russian software firms - 1C Company.

We have been asked to predict total sales for every product and store in the next month.

## Project structure

.
├── README.md
├── archives
├── checkpoints
│ ├── frozen_data.npz
│ └── xgboost
│ ├── model.pkl
│ └── model_0.pkl
├── data
│ ├── archive
│ │ └── competitive-data-science-predict-future-sales.zip
│ ├── item_categories.csv
│ ├── items.csv
│ ├── sales_train.csv
│ ├── sample_submission.csv
│ ├── shops.csv
│ └── test.csv
├── debug.log
├── environment.yml
├── mlruns
│ └── 0
├── notebooks
│ ├── 01-baseline-pre-processing.ipynb
│ └── Playground.ipynb
├── outputs
│ ├── xgboost_submission.csv
│ └── xgboost_submission_0.csv
├── pyproject.toml
└── src
├── config
│ └── model_params.py
├── dispatcher.py
├── feature_generator.py
├── main.py
├── mlruns
│ └── 0
│ └── meta.yaml
├── settings.py
└── utils.py

## Run the project

```
python -m src.main
```
