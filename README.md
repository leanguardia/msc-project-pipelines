# Pipes - MSc Advanced Project

**Pipes** is an end-to-end Machine Learning (ML) system implemented as supporting evidence for the practice-based research called  *The Data Prepararion Pattern*. A masters dissertation final project of the masters course *Advanced Computing - Machine Learning, Data Mining and High Performance Computing (MSc)* at the University of Bristol.

The research endeavour consisted in a theoretical and empirical study of ML systems good implementation practices with focus in **Data Preparation**, built on two traditional software engineering elements; design pattterns and software testing.

## Overview
Pipes consists in four end-to-end pipelines, two regressions and two classifications. These contain all stages of the ML workflow from data integration, across data cleaning, data validation, feature engineering, model training, evaluation and inference. Prior to pipeline 

## Architecture
Inspired by Yokahoma's layered ML system architecture.

IMG

## Contribution
The Data Preparation Pattern as a generalizable template structure to guide flexible and reliable implementation of all activities after data integration, and before modelling in the ML workflow. The patter abstracts all included activities in data clearning, data validation and feature engineering.

IMG

#### Folder Structure
- app: Flas web application
- models: 
- pipelines
- test
- *.ipynb

## Dependencies 
- Python 3
- pandas
- numpy
- jupyter (notebook)
- scikit-learn
- flask
- flask_sqlalchemy
- sqlalchemy
- pytest

## Set up
```
> from app import db
> db.create_all()
```

## Run Unit-Tests
```
> python -m pytest

```