# Explore

# Dataframe manipulations
import pandas as pd
import numpy as np

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy.stats import pearsonr

# Sklearn
from sklearn.preprocessing import MinMaxScaler


def linear_tests(train, conts, target):
    '''
    This function takes our train dataframe and a list of continous variables and outputs a jointplot, lmplot, and pearsonr hypothesis test.
    There is no return but there is a print statement of whether the null hypothesis is rejecting or accepted.
    '''
    # Looping through the chosen continuous variables
    for col in conts:
        # Creation of a jointplot using our target variable and the selected column
        sns.jointplot(y=target, x=col, data=train, kind='scatter')
        plt.xlabel(f'{col}')
        plt.ylabel(f'{target}')
        plt.show()
        # Creation of a lmplot using our target variable and the selected column
        sns.lmplot(x=col, y=target, data=train, scatter=True, hue=None, col=None)
        plt.xlabel(f'{col}')
        plt.ylabel(f'{target}')
        plt.title(f'{col} by {target}')
        plt.show()
        # Printing of our hypothesis surrounding the linear relationship between column and target variable
        print(f'H0: There is no linear relationship between {col} and {target}.')
        print(f'HA: There is a linear relationship between {col} and {target}.')
        print('----------------------------------------------------------------')
        # Established alpha
        alpha = .05
        # Pearsonr test to determine r and p values
        r, p = pearsonr(train[col], train[target])
        # Analysis of p value for rejection or acceptance of null hypothesis
        if p < alpha:
            print(f'p-value: {p}\n')
            print('With a p-value below our established alpha we reject the null hypothesis.')
        else:
            print('We fail to reject the null hypothesis.')


def cat_visuals(train, cats, target):
    '''
    This function takes our dataframe, a list of categorical variables, and our target variable and makes
    a countplot and stripplot from them. Returns nothing.
    '''
    # Looping through the chosen categorical variables
    for col in cats:
        # Creation of a countplot using our target variable and selected column
        sns.countplot(y=col, data=train)
        plt.xlabel(f'Count of {col}')
        plt.ylabel(f'{col}')
        plt.title(f'{col} (count)')
        plt.show()
        # Creation of stripplot 
        sns.stripplot(x=col, y=target, data=train)
        plt.xlabel(f'{col}')
        plt.ylabel(f'{target}')
        plt.title(f'{target} by {col}')
        plt.show()


def heatmap_zillow(train):
    '''
    This function takes in our train dataframe and creates a correlation from it. This correlation is then mapped as a heatmap.
    '''
    # Creation and mapping of the correlation
    corr = train.corr()
    # Separating the top half of the heatmap to laster mask
    matrix = np.triu(corr)
    plt.figure(figsize=(16,9))
    ax = sns.heatmap(corr, cmap='coolwarm', mask=matrix)
    ax.set(title='Heatmap')


def scale_zillow(train, validate, test, quants):
    '''
    This function takes train, validate, and test dataframes and scales their numerical columns that are not
    the target variable. The scaler is fit on the train and then transformed on all three dataframes. Returns the 
    three dataframes.
    '''
    # Creation of scaler
    scaler = MinMaxScaler()
    # Fit scaler to train
    scaler.fit(train[quants])
    # Apply to train, validate, and test dataframes
    train[quants] = scaler.transform(train[quants])
    validate[quants] = scaler.transform(validate[quants])
    test[quants] = scaler.transform(test[quants])
    # Return the three scaled dataframes
    return train, validate, test