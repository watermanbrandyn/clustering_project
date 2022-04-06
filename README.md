# Predicting Tax Value, Zillow (Regression Project)

## Table of Contents
- [Project Goal](#project-goal)
- [Project Description](#project-description)
- [How to Reproduce](#how-to-reproduce)
- [Initial Questions](#initial-questions)
- [Data Dictionary](#data-dictionary)
- [Project Plan](#project-plan)
   - [Wrangling](#wrangling)
      - [Acquire](#acquire)
      - [Preparation and Splitting](#preparation-and-splitting)
  - [Exploration](#exploration)
  - [Modeling](#modeling)
  - [Deliverables](#deliverables)
    - [Final Report](#final-report)
    - [Modules](#modules)
    - [Predictions](#predictions)
- [Summary and Recommendations](#summary-and-recommendations)

## Project Goal
The goal of this project is to offer analysis to Zillow that can help predict the tax values ($) for Single Family Properties. 
This will be done by identifying some of the key attributes that drive tax value, creating models to help predict this value (using these key attributes), and offering recommendations to assist Zillow in making these predictions moving forward.  

## Project Description
There are a number of factors that can influence the tax value of a property. These can range from physical attributes of the property, individual actions of property owners (such as remodeling choices), or even the decisions made by local, state, or federal entities. This complex web of interactions can certainly make for a difficult task when trying to predict a tax value, but the ability to do so will be greatly beneficial to both Zillow and their clients. 

Through the identification of key drivers, application of these drivers to models, and the testing of prediction capabilities we can help Zillow obtain an advantage in the acquistion and sales of real-estate. This in turn can help Zillow attain a reputation of having a better understanding of the volatility behind pricing, and ensure the company remains a leader in both consultation regarding, and movement of, the housing marketplace. 

## How to Reproduce 
To reproduce the outcomes in this project:
1. Have an env.py file with credentials (hostname, username, password) to access a SQL database that contains Zillow data. Codeup's 'zillow' data was utilized
   for this project. 
2. Clone this repo and ensure you have all of the necessary modules and notebooks. Confirm that the .gitignore includes your env.py file to secure credentials.
3. Use of these libraries: pandas, numpy, matplotlib, seaborn, sklearn.
4. Be able to run the 'Final Report' jupyter notebook file. 
   - Supplemental regression workbooks may also be useful in identifying some of the steps taken prior to the cleaner final code 

## Initial Questions
_What is our minimum viable product? (MVP)_
- At a minimum this project aims to provide:
   - An explanation of several key drivers of tax value.
   - At least one model that can perform better than a baseline prediction of tax value.
   - At least two recommendations for Zillow on how to predict tax value moving forward.

_Initial Data Centric Questions_
- How do we define 'Single Family Properties'? 
- What is our best method for a baseline prediction of tax value (mean, median, mode)?
   - What is our baseline prediction of tax value?
- How do our core physical attributes impact tax value?
   - Square feet, Bedrooms, Bathrooms 
- How do our add-on components impact tax value?
   - Examples: Pools, Decks, Type of Heating/Cooling, etc...
- How do immutable attributes impact tax value?
   - Examples: Geographic location (State, County, Zipcode, Neighborhood), Year Built, etc...

_Initial Business Questions_
- What are the main drivers and which are not as important in tax value predictions?
- In what ways can this knowledge benefit Zillow and their clients? 

_Initial Hypotheses_
- Is there a linear relationship between tax value and our core physical attributes? (Three separate Pearson investigations)
- Is there a difference in tax values by geographic attributes? (Various T-Tests/ANOVA investigations)
- Is there a difference in tax values and having add-on components? (Various T-Tests)

## Data Dictionary
| Attribute                             | Definition                                        | Data Type | Additional Info     |
|:--------------------------------------|:--------------------------------------------------|:---------:|:--------------------|
| bedrooms                              | Number of bedrooms                                | Float     | Scaled              |
| bathrooms                             | Number of bathrooms                               | Float     | Scaled              |
| house_area                            | Square feet of house                              | Float     | Scaled              |
| lot_area                              | Square feet of lot                                | Float     | Scaled              |
| tax_value                             | House value for tax purposes                      | Float     | Target variable     |
| age                                   | Age of house                                      | Float     | Scaled              |
| fips                                  | Federal Information Processing Standards (county) | uint8     | Three unique values |

## Project Plan 
This project will start with some initial planning and question exploration before we even access the data. The question exploration has been delved out in the _Initial Questions_ section. 
Additionally let us detail what is to be provided at the conclusion of this project:
 - This README.md
 - Final Report.ipynb 
 - Workbooks and modules used

Moving forward we will **wrangle (acquire/prepare)** our data, **explore** for insights on key drivers, create **models** for prediction, and apply the best ones for the purpose of curating some **predictions**. This will all be **summarized** and **recommendations** for Zillow will be provided. 
For a more detailed breakdown of these steps please see the Final Report and workbooks provided. 

### Wrangling 
This section contains our acquisition and preparation of the data.
#### Acquire
The acquire.py file contains the code that was used for acquiring the 'zillow' data. There is a **get_db_url()** function that is used to format the credentials for interacting with a SQL server, and the **acquire_zillow()** function that queries the SQL server for the data. For this project Codeup's 'zillow' SQL database was used. The env.py file used, and the credentials within, are not included in this project and as covered under _How To Reproduce_ must be curated with one's own information.

#### Preparation and Splitting
The prepare.py file contains the code that was used for preparing the data. The **prepare_zillow()** function takes the acquired dataframe and cleans it for our exploratory purposes. For this dataset this includes the use of a **remove_outliers()** function, along with a **zillow_split()** function that provides our train, validate, and train dataframes. Although not part of the initial prepare function the **scale_zillow()** function is utilized to scale the data prior to modeling and for the later stages of exploration. 

### Exploration 
For exploration we used only our train dataframe. The explore.py file contains a number of functions that were used to help gain insights into our data, using both visual and statistical methods. We delved out the key factors shown to impact tax value and our train, validate, and test dataframes only include these features. The main takeaways from exploration are that tax value is influenced by:
- bedrooms
- bathrooms
- house_area
- lot_area
- age
- fips

However, the biggest issue found with the Zillow data generally is that a lot of features that could possibly be deemed important were full of too many nulls to be useful in this project.

### Modeling
We created a number of models that included Ordinary Least Squares (OLS), Lasso & Lars, Polynomial Regression (using LinearRegression), and a Generalized Linear Model (GLM, using TweedieRegressor) types using our selected feature sets. Our report covers the top three performing models with the best performance being the TweedieRegressor. From this model we obtained a **12%** improvement over the baseline, with a **$28,851.55** difference in prediction values. While this does perform better than not having a model, it is not a substantial enough improvement to recommend use for real world applications.

### Deliverables
The main deliverable from this project are the Final Report. Additionally there are modules that contain the functions used and workbooks where a deeper exploration of the process can be seen.

#### Final Report
The Final Report can be ran to reproduce the same results from start to finish. 

#### Modules
The modules included in this project are:
- acquire.py
- prepare.py
- explore.py
- modeling.py

#### Predictions
The modeling.py module could be used/modified to show predictions and contains functions that alter the train, validate, and test dataframes to store the outcomes from the models. More specifically the y component (target variable) has the predictions added to their respective dataframes.

### Summary and Recommendations
We were successful at identifying some key drivers that influence tax value, but do not feel that these features alone are enough to create a useful model. The source data has far too many nulls to make use of features that could possibly be deemed important. Among these are all of the add-on components of a house (such as pool, type of cooling/heating, etc...) or more specific locational data that are known to be important in determining the value of a house. 

Moving forward the recommendation is that Zillow work on fixing their gaps in data prior to attempts at creating useful models out of it. If this is not a possibility one could spend a large amount of time trying to impute, modify, or curate those gaps for them but this does not seem like the best route to take. In this instance data acquisition, at the source, is causing a bottle-neck in terms of possible utilities and opportunities.



