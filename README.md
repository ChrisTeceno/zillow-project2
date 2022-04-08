# zillow-project2

# PROJECT OVERVIEW

The goal of this project is to **find drivers of logerror** and **create a predictive model** that can predict the logerror of single family properties based on the features of the home. The model is trained on data from zillow found on kaggle and looks specifically at transactions for the year 2017.

    Note: logerror is the log of the difference between the actual sale price and the estimated price.

## Project Description

Zillow attempted with varying degrees of success to predict logerror based of off several data points. We will use a smaller subset of that data to train and test our models and eventually present the best performing model. Improvements and insights could then be refactored into larger scale models.

## Goals

1. Find the drivers of logerror. 
2. Make a model that performs better than baseline. 
3. Refine or create a better model. 
4. Give recommendations for new data to create better models.

## Aquiring Data

Data is store in codeups SQL database and a SQL query is run to pull the data. This is done using the wrangle_zillow.py module. wrangle_zillow.get_zillow_data() houses the SQL query and returns a pandas dataframe. This is then prepped by wrangle_zillow.py to be used in the exploration and modeling.

wrangle_zillow.wrangle_zillow() is the main function that calls the relevant functions to get, prep, and clean the data.

## Preparing Data

The data actually gets cleaned and aquired in one function call above. Some things to note about the preparation:
1. Nulls are removed because they are less than 1% of the data.
2. duplicate parcelIDs are removed.
3. The data is cleaned by removing outliers.
4. single units are selected based on unit count or landusetypeid if unitcount is missing.
5. missing data is filled if possible.
6. Transaction date is converted to q1, q2, q3, q4 and dummy variables are made.
7. Propertycountylandusecode is converted to a dummy variable.
8. Yearbuilt is converted to age of home based on 2017.
9. repetive features are removed as well as features with too many missing values.
10. LA county ends up being the only county with sufficent data. More models could be created to explore other counties.

This is also stored in .csv file for later use.

### Split the data

Data is split into train, validate, and test sets to prevent overfitting and to allow for data to be used to evaluate the model.

### Scale the data
The MinMaxScaler is first trained on the training data and then used to scale the data. The target is not scaled

## Initial questions
### Question 1:
    
    Is there a correlation between log error and bathroom count?

## Question 2:
    
    Is there a correlation between log error and latitude?

## Question 3:

    Is there a correlation between log error and longitude?

## Question 4:

    Is there a correlation between log error and age?

### Data Dictionary

| Variable                   | Meaning                                                                  
| -----------                | -----------                                                                        
| bedroomcnt                 | Number of bedrooms in home                                                
| bathroomcnt                | Number of bathrooms in home including fractional bathrooms                    
| age                        | Age of the structure (in years) at the time the data was collected (2017) 
| unitcount                  | Number of units in the home
|'buildingqualitytypeid'     | Type of building quality (1 = typical, 2 = above average, 3 = very good, 4 = excellent)
|calculatedfinishedsquarefeet| Calculated total finished living area of the home (in square feet)
|heatingorsystemtypeid       | Type of heating system (1 = gas, 2 = electric, 3 = forced air, 4 = hot water, 5 = steam, 6 = wood, 7 = other)
|latitude                    | Latitude of the property
|longitude                   | Longitude of the property
|lotsizesquarefeet           | Calculated total finished living area of the home (in square feet)
|rawcensustractandblock      | Census tract and block in which the property is located
|regionidcity                | City in which the property is located
|regionidzip                 | Zip code in which the property is located
|structuretaxvaluedollarcnt  | Tax value of the structure (in dollars)
|taxvaluedollarcnt           | Tax value of the property (in dollars)
|taxamount                   | Tax amount (in dollars)
|censustractandblock         | Census tract and block in which the property is located
|q3 transaction date         | Quarter of the year the transaction was made (1 yes, 0 no)
|q4 transaction date         | Quarter of the year the transaction was made (1 yes, 0 no)
|propertycountylandusecode   | County-land use code in which the property is located




### Steps to Reproduce
1. Download the data from CodeUp SQL database using query in acquire.py
2. Clone all modules from my github repo
3. Ensure all libraries are installed, sklearn, pandas, numpy, matplotlib, seaborn, and scikit-learn

## Plan
1. Get the data
2. clean and prepare the data, remove outliers, scale, dummy variables
3. split the data into training, validate and testing sets
4. explore the training data. Look for features to use in modeling, make visuals
5. build a baseline model
6. build own models and compare
7. test best model against test set
8. give recomedations to move forward

## CONCLUSION
### Drivers of logerror
'Propertycountylandusecode', 'calculatedfinishedsquarefeet', 'landtaxvaluedollarcnt', 'taxamount' are the most important drivers of logerror.

### Model improvement
Our models almost all perform better than the baseline model. With one having a RMSE of 0.001 better than the baseline. I would like to try different ways to scale the data and remove outliers, and reconfigure clusters

I would also like to look at making different models for each cluster. A cluster with k values would have k models with the results aggregated.

### Recommendations
More features I would add in would be proximity to important locations such as hospitals, schools, police departments, beaches, as well as crime rates.