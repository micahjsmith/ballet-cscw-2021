# census-model

## Task description

This task is to create a feature engineering pipeline and a model that can be used to predict personal income from raw survey responses to the U.S. Census's [American Community Survey](https://www.census.gov/programs-surveys/acs/) (ACS). The end-to-end model can then be used to optimize administration of the survey, direct public policy interventions, and assist empirical researchers.

Your task is as follows:

* create a feature engineering pipeline and a model for the prediction task described [in this document](#prediction-target)
* implement the script `predict.py` that can train a model on the fixed training data described below and then make batch predictions on any new, unseen raw survey responses.

You can use any framework, development environment, and project organization that you want. *You should not focus on making your implementation fast or well-structured* as the only requirement is that the `predict.py` script works to make new predictions. You may use git to track your changes at any point, but when you have completed the task you should *commit all code files, data files, and notebooks* and push them to this repository (even if you would not otherwise track all of these files).

## Data

### Input data

The input data is the raw survey responses to the 2018 US Census American Community Survey (ACS). This is known as the "Public Use Microdata Sample" because otherwise most numbers from the ACS are reported in aggregate.

* The data documentation can be viewed [here](https://9mkmatu5ijhtfvj1zmz.s3.amazonaws.com/census/ACS2018_PUMS_README.pdf)
* The data dictionary can be viewed [here](https://9mkmatu5ijhtfvj1zmz.s3.amazonaws.com/census/PUMS_Data_Dictionary_2018.pdf) in PDF form, or [here](https://9mkmatu5ijhtfvj1zmz.s3.amazonaws.com/census/PUMS_Data_Dictionary_2018.csv) in CSV form.
* The dataset is created by merging the "household" and "person" parts of the survey. Thus one row of the dataset contains the responses for one person to both the household and person surveys. A person is identified by a unique `SERIALNO`. A set of "reasonable" rows is filtered as follows: (1) individuals older than 16 (2) personal income greater than $100 (3) hours worked in a typical week greater than 0.

The resulting training dataset has 30085 rows (people) and 494 columns (raw).

### Input labels

The "raw labels" in the training data are the set of all columns from the raw ACS responses that contain income data. We care about personal income (`PINCP`) which is one of these columns. You may use these columns however you like during the training process. However, for evaluation, you should use the exact prediction target described in the next section.

### Prediction target

The prediction target is whether an individual respondent will earn more than $84,770 in 2018. Though a bit contrived, this comes from adapting the classic ML ["census"](https://archive.ics.uci.edu/ml/datasets/Census+Income) dataset to the modern era. The original prediction target is to

> determine whether a person makes over 50K a year.

Thus we adjust for inflation from 1994 to 2018.

You can use the following snippet to get the prediction target from the raw labels:

```python
# y_df is the raw labels
target = (y_df['PINCP'] > 84770).astype(int)
```
