import acquire
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os
from scipy import stats
import numpy as np


def split_data(df, y_value, stratify=True):
    """ General use function to split data into train and test sets. 
    Stratify = True is helpful for categorical y values"""
    # split the data set with stratifiy if True
    if stratify:
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[y_value]
        )
        train, validate = train_test_split(
            train, test_size=0.3, random_state=42, stratify=train[y_value]
        )
    else:  # if stratify is false (for non-categorical y values)
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        train, validate = train_test_split(train, test_size=0.3, random_state=42)
    return (train, validate, test)


def split_x_y(df, y_value):
    """split data into x and y"""
    x = df.drop(columns=[y_value])
    y = df[y_value]
    return x, y


def prep_zillow(use_cache=True):
    """pull from full zillow.csv unless prepped_zillow.csv exists, 
    includes option to overide using csv"""
    filename = "prepped_zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Prepped csv exist, pulling data...")
        return pd.read_csv(filename)

    print("preparing data from get_zillow_data()")
    df = acquire.get_zillow_data()
    # drop the nulls, its a small subset of data so its ok to drop
    # if nulls made up a larger percentage of the data, we could impute data if possible
    df = df.dropna()
    # convert year built to age
    df["age"] = 2017 - df["yearbuilt"]
    # drop the yearbuilt column
    df = df.drop(columns=["yearbuilt"])
    # convert fips to county name
    df["fips"] = df["fips"].replace(
        {6037.0: "la_county", 6059.0: "orange_county", 6111.0: "ventura_county"}
    )
    # make dummy columns for fips
    df = pd.concat(
        [
            df,
            (
                pd.get_dummies(
                    df[["fips"]],
                    dummy_na=False,
                    drop_first=False,  # i did not drop first to make it more human readable
                )
            ),
        ],
        axis=1,
    )
    # drop fips column
    df.drop(columns=["fips"], inplace=True)
    # save to csv
    print("Saving prepped data to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def remove_outliers(df, z_score=3):
    """remove outliers from data based on z score"""
    # return df if the z score is less than the threshold chosen
    return df[(np.abs(stats.zscore(df)) < z_score).all(axis=1)]


def scale_features(
    X_train,
    X_validate,
    X_test,
    columns=["bedroomcnt", "bathroomcnt", "calculatedfinishedsquarefeet", "age"],
):
    """scale features using RobustScaler"""
    # add 'scaled' to the column names in columns_to_scale
    scaled_columns = ["scaled_" + col for col in columns]
    # make and fit the scaler
    scaler = RobustScaler().fit(X_train[columns])
    # Use the scaler to transform train, validate, test (use the thing)
    X_train_scaled = scaler.transform(X_train[columns])
    # put scaled columns into df
    X_train_scaled = pd.DataFrame(
        X_train_scaled, index=X_train.index, columns=scaled_columns
    )
    # concat the scaled df back onto original
    X_train = pd.concat([X_train, X_train_scaled], axis=1)
    # do the same for validate and test
    X_validate_scaled = scaler.transform(X_validate[columns])
    X_validate_scaled = pd.DataFrame(
        X_validate_scaled, index=X_validate.index, columns=scaled_columns
    )
    X_validate = pd.concat([X_validate, X_validate_scaled], axis=1)
    X_test_scaled = scaler.transform(X_test[columns])
    X_test_scaled = pd.DataFrame(
        X_test_scaled, index=X_test.index, columns=scaled_columns
    )
    X_test = pd.concat([X_test, X_test_scaled], axis=1)
    return X_train, X_validate, X_test
