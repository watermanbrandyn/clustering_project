# Predicting Log Error, Zillow (Clustering Project)

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
  - [Clustering](#clustering)
  - [Modeling](#modeling)
  - [Deliverables](#deliverables)
    - [Final Report](#final-report)
    - [Modules](#modules)
    - [Predictions](#predictions)
- [Summary and Recommendations](#summary-and-recommendations)

## Project Goal
The goal of this project is to offer analysis to Zillow that can help predict the log error (Zestimate error) for Single Unit house values. 
This will be done by identifying some of the key attributes that drive log error, creating models to help predict this value (using these key attributes), and offering recommendations to assist Zillow in making these predictions moving forward.  

## Project Description
A large part of Zillow's current utility in the housing market revolves around the ability to accurately predict the price of a house. This is essential to not just the consultation function that Zillow provides, but also for opportunities that may arise where the company can actively participate in the buying and selling of the market. As a leader in this space it is critical that Zillow refine their processes to produce as little error as possible when stating a house value.

Through the identification of key drivers, application of these drivers to models, and the testing of prediction capabilities we can help Zillow obtain an advantage in understanding the mechanisms behind the price of a house. If Zillow produces a better understanding of the volatility behind pricing, it will ensure that the company remains a leader in both consultation regarding, and movement of, the housing marketplace. 

## How to Reproduce 
To reproduce the outcomes in this project:
1. Have an env.py file with credentials (hostname, username, password) to access a SQL database that contains Zillow data. Codeup's 'zillow' data was utilized
   for this project. 
2. Clone this repo and ensure you have all of the necessary modules and notebooks. Confirm that the .gitignore includes your env.py file to secure credentials.
3. Use of these libraries: pandas, numpy, matplotlib, seaborn, sklearn.
4. Be able to run the 'Final Report' jupyter notebook file. 
   - Supplemental workbooks may also be useful in identifying some of the steps taken prior to the cleaner final code 

## Initial Questions 
_Initial Data Centric Questions_
1. Do primary house attributes impact log error? (bedrooms, bathrooms, age, squarefeet)
2. Do secondary house attributes impact log error? (num_fireplace, threequarter_baths, hottub_or_spa, has_pool)
3. Does geography impact log error? (latitude, longitude, regionidzip, fips)
4. Can we successfully use any of our features to cluster for log error predictions?
    - Geographic clustering
        - Latitude/Longitude
    - Continuous feature clustering
5. What can we identify about the data when log error is positive or negative?

_Initial Hypotheses_ 
1. Is there a linear relationship between log error and our continuous features? (Pearsonr)
2. Is there a difference in the mean log error for selected subsets and the entire dataset? (one-sample t-test)
3. Is there a difference in the mean log error of particular zip codes and the entire dataset? (one-sample t-test)

## Data Dictionary
| Attribute                             | Definition                                             | Data Type | Additional Info                 |
|:--------------------------------------|:-------------------------------------------------------|:---------:|:--------------------------------|
| bathrooms                             | Number of bathrooms                                    | Float     | Scaled                          |
| bedrooms                              | Number of bedrooms                                     | Float     | Scaled                          |
| squarefeet                            | Total Squared Feet of house                            | Float     | Scaled                          |
| num_fireplace                         | Number of fireplaces                                   | Float     | Scaled                          |
| latitude                              | Geodata: latitude                                      | Float     | Clustered                       |
| longitude                             | Geodata: longitude                                     | Float     | Clustered                       |
| threequarter_baths                    | Number of three quarter bathrooms                      | Float     | Scaled                          |
| logerror                              | Percent error of Zestimate                             | Float     | Target Variable                 |
| age                                   | Age of house                                           | Float     | Scaled                          |
| fips                                  | Federal Information Processing Standards (county)      | uint8     | Categorical (3 unique in model) |
| hottub_or_spa                         | If house has a hottub or spa                           | uint8     | Categorical (1/0)               |
| has_pool                              | If house has a pool                                    | uint8     | Categorical (1/0)               |
| tax_delinquency                       | If house is delinquent on taxes                        | uint8     | Categorical (1/0)               |
| regionidzip                           | Zip code house resides in                              | uint8     | Categorical (39 unique in model)|
| age_sqft_cluster                      | Cluster made with age and squarefeet                   | uint8     | Categorical (3 unique in model) |
| conts_cluster                         | Cluster made with bathrooms, bedrooms, age, squarefeet | uint8     | Categorical (3 unique in model) |
| lat_long_cluster                      | Cluster made with latitude and longitude               | uint8     | Cateogircal (1 unique in model) |

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
The wrangle_zillow.py file contains the code that was used for acquiring the Zillow data. There is a **get_db_url()** function that is used to format the credentials for interacting with a SQL server, and the **acquire_zillow()** function that queries the SQL server for the data. For this project Codeup's 'zillow' SQL database was used. The env.py file used, and the credentials within, are not included in this project and as covered under _How To Reproduce_ must be curated with one's own information.

#### Preparation and Splitting
The wrangle_zillow.py file contains the code that was used for preparing the data. The **prepare_zillow()** function takes the acquired dataframe and cleans it for our exploratory purposes. To accomplish this a number of functions from the module are used. Nulls and missing values are identified and removed or imputed as necessary. Outliers are removed to make our work more widely usable, and columns are modified to their appropriate data types or formats. Our intitial dataframe is then split into train, validate, and test splits. 

### Exploration
For exploration we used only our train dataframe. The explore.py file contains a number of functions that were used to help gain insights into our data, using both visual and statistical methods. We delved out the key factors shown to impact log error and our train, validate, and test dataframes only include these features. 

#### Clustering
A large component of our exploration was the use of clustering to help identify key drivers of log error. Clustering on geographical and continuous features provided insights into which clusters are most impactful, and allowed for the dataframes to be further trimmed to only the most optimal features to use for modeling. 

The main takeaways from exploration are that log error is influenced by: 
- bathrooms
- bedrooms
- squarefeet
- num_fireplace
- threequarter_baths
- logerror (target)
- age
- has_pool 
- tax_delinquency
- lat_long_cluster (cluster based on latitude and longitude)
- conts_cluster (cluster based on bathrooms, bedrooms, squarefeet, and age)
- age_sqft_cluster (cluster based on age and squarefeet)
- regionidzip (39 unique encoded zip codes)

### Modeling 
We created a number of models that included Ordinary Least Squares (OLS), Lasso & Lars, Polynomial Regression (using LinearRegression), and a Generalized Linear Model (GLM, using TweedieRegressor) types using our selected feature sets. Showing the result of all four, the OLS and Tweedie models performed nearly identical, and the Tweedie was selected for use with the test dataframe since it performed the best previously on a regression model to find tax value for a home. Our test ended up performing worse than the baseline, and ultimately only the train data for all of the models beat the baseline. None of the validate data did. 

### Deliverables 
The main deliverable from this project are the Final Report. Additionally there are modules that contain the functions used and workbooks where a deeper exploration of the process can be seen.

#### Final Report
The Final Report can be ran to reproduce the same results from start to finish. 

#### Modules
The modules included in this project are:
- wrangle_zillow.py
- explore.py
- modeling.py

#### Predictions
The modeling.py module could be used/modified to show predictions and contains functions that alter the train, validate, and test dataframes to store the outcomes from the models. More specifically the y component (target variable) has the predictions added to their respective dataframes.

### Summary and Recommendations
Ultimately we were not successful in identifying drivers that were capable of creating a useful, or even superior, regression model to that of a baseline guess. While we were able to gather features that were deemed statistically significant enough to use for modeling, they were not able to result in a model that could be utilized by Zillow. 

The first recommendation I would provide is to gather better, or correct the available data. There are a lot of issues with the utility of the data as it currently sits on the SQL server, but utltimately new features are needed to provide a model that obtains the goals set out in the project. While house features may contribute enough to the value of a house to be used successfully to predict value (Regression Project prior to this project), they are not useful enough in predicting the log error produced by Zillow for predicting said value. Perhaps for this data that centers around the 'hidden' bonuses of a house should be gathered, such as proximity to schools, quality of nearby businesses, and if there is or is not an HOA. (As some examples)

Moving forward one could spend more time with the data that is currently available but a better solution is probably to work to obtain features that are not currently present. 



