# Housing Price & Assessment Bias

Linear-regression pipeline that predicts Cook County home sale prices and measures where the model over- and under-estimates across price bands.

## Problem

Given 200K+ Cook County Assessor property records, predict sale price and check whether the errors are systematic across price bands. That pattern is what documented inequities in property-tax assessment look like in practice. Built as a course project for a Data Science course at UC Berkeley.

## Approach

- Engineered 15+ features: log-transformed square footage, polynomial age and geographic terms, latitude-by-longitude interactions, and one-hot road and garage features.
- Filtered non-arm's-length $1 sales so the training data mirrored real assessment practice.
- Trained a linear regression model and evaluated it with 4-fold cross-validation.
- Broke prediction error down by price band to quantify systematic over- and under-estimation, and tied modeling choices back to the assessment-equity question.

## Result

Test RMSE of **0.562** under 4-fold cross-validation, plus a price-band error analysis showing where the model systematically over- and under-estimates.

## Tools

Python, pandas, NumPy, scikit-learn.

## Code availability

This project was completed as coursework. Per course policy, solution code isn't posted publicly here. I'm glad to walk through the implementation directly, reach out at vdkarthikeya@berkeley.edu.
