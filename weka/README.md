# WEKA Practical Guide

This repository includes `dataset/hostel_complaints.arff`, which is ready to open in WEKA.

## Assignment 4 – Preprocessing
Explorer -> Preprocess
- Open hostel_complaints.arff
- Apply ReplaceMissingValues if needed
- Apply Normalize to numeric attributes
- Apply Discretize to convert numeric fields into ranges

## Assignment 5 – Association Rules
Explorer -> Associate -> Apriori -> Start

## Assignment 6 – J48
Explorer -> Classify -> trees -> J48
- Set `Priority` as the class attribute
- Use 10-fold cross-validation

## Assignment 7 – Naive Bayes
Explorer -> Classify -> bayes -> NaiveBayes
- Set `Priority` as the class

## Assignment 8 – Regression
Explorer -> Classify -> functions -> LinearRegression
- Set `Resolution_Time_Days` as the target

## Assignment 9 – Clustering
Explorer -> Cluster -> SimpleKMeans
- Use 3 clusters
- Select numeric/processed attributes

The Flask application uses Python equivalents so it can run without requiring a local WEKA installation.
