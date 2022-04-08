import pandas as pd
import os
from env import get_db_url
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# get_zillow_data while maintaining my original query from previous project
def get_zillow_data(use_cache=True):
    """pull from SQL unless zillow.csv exists"""
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("zillow")
    query = """
        SELECT prop.*, 
       pred.logerror, 
       pred.transactiondate, 
       air.airconditioningdesc, 
       arch.architecturalstyledesc, 
       build.buildingclassdesc, 
       heat.heatingorsystemdesc, 
       landuse.propertylandusedesc, 
       story.storydesc, 
       construct.typeconstructiondesc 

FROM   properties_2017 prop  
       INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                   FROM   predictions_2017 
                   GROUP  BY parcelid, logerror) pred
               USING (parcelid) 
       LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
       LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
       LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
       LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
       LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
       LEFT JOIN storytype story USING (storytypeid) 
       LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
WHERE  prop.latitude IS NOT NULL 
       AND prop.longitude IS NOT NULL AND transactiondate <= '2017-12-31' 
"""

    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def clear_parcel_id_duplicates(df):
    """remove duplicates from parcelid column"""
    df = df.drop_duplicates(subset="parcelid")
    return df


# return a df with number of missing values for each column and the percentage of missing values
def missing_values_table(df):
    """return a df with number of missing values for each column and the percentage of missing values"""
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )
    print(
        "Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values."
    )
    return mis_val_table_ren_columns

    # return a df with number of missing columns and the percentage of missing values


def missing_columns_table(df):
    """return a df with number of missing columns and the percentage of missing values"""
    df2 = pd.DataFrame(
        df.isnull().sum(axis=1), columns=["num_cols_missing"]
    ).reset_index()
    # groupby num_cols_missing and count the number of rows with the same num_cols_missing
    df2 = df2.groupby("num_cols_missing").count().reset_index()
    # rename index to num_rows
    df2 = df2.rename(columns={"index": "num_rows"})
    # add column with percentage of missing values
    df2["percent_missing"] = (100 * df2["num_cols_missing"] / df.shape[1]).round(1)
    return df2.sort_values("num_cols_missing", ascending=False)


def get_single_units(df):
    """returns a df with only single units"""
    # list of property types that are likely single unit
    likely_single_unit = [261, 262, 263, 264, 265, 266, 268, 273, 275, 276, 279]
    # keep only single unit properties by property type
    temp_df = df[df.propertylandusetypeid.isin(likely_single_unit)]
    # keep only single unit properties by unitcnt ignoring null
    return temp_df[(temp_df["unitcnt"] < 2) | (temp_df["unitcnt"].isnull())]


def handle_missing_values(df, prop_required_col=0.7, prop_required_row=0.7):
    """drop column/row if it has more than prop_required_row missing values or more than prop_required_col missing values"""
    df2 = df.copy()
    threshold = int(round(prop_required_col * len(df2.index), 0))
    df2.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df2.columns), 0))
    df2.dropna(axis=0, thresh=threshold, inplace=True)
    return df2


def drop_replace_nulls(df):
    """drop or replace null values"""
    # drop rows where yearbuilt is null
    df.dropna(subset=["yearbuilt"], inplace=True)
    # drop rows where regionidzip is null
    df = df[df.regionidzip.notnull()]  # remove the 43 out of 72407 rows
    # fill missing values in lotsizesquarefeet with median for same regionidzip and yearbuilt
    df["lotsizesquarefeet"] = df.groupby(["regionidzip", "yearbuilt"])[
        "lotsizesquarefeet"
    ].transform(lambda x: x.fillna(x.median()))
    # fill missing values in regionidcity with median for same regionidzip and yearbuilt
    df["regionidcity"] = df.groupby(["regionidzip", "yearbuilt"])[
        "regionidcity"
    ].transform(lambda x: x.fillna(x.median()))
    # drop the remaining null values
    df.dropna(inplace=True)
    return df


def split_data(df, y_value, stratify=False):
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


def convert_fips(df):
    """convert fips to string then dummy encode"""
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
    return df


def detect_outliers(df, col, k=1.5):
    """look for outliers in a column of a dataframe using IQR, k"""
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * k)
    upper_bound = q3 + (iqr * k)
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]


def remove_outliers(df, k=1.5):
    """remove outliers from all quantitative variables"""
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df = df[~df[col].index.isin(detect_outliers(df, col, k).index)]
    return df


def make_dummies(df, col):
    """make dummy columns for a column in a dataframe"""
    df = pd.concat(
        [df, pd.get_dummies(df[col], dummy_na=False, drop_first=True)], axis=1
    )
    return df


def bin(df, col, bins):
    """bin a column of a dataframe"""
    df[col] = pd.cut(df[col], bins, labels=False)
    return df


def handle_transactiondate(df):
    """'convert transactiondate to datetime then string quarter then dummy encode"""
    df = df.copy()
    df["transactiondate"] = pd.to_datetime(df["transactiondate"])
    df["transactiondate"] = df["transactiondate"].dt.quarter
    # add 1 to the transactiondate column
    df["transactiondate"] = df["transactiondate"] + 1
    # convert to string
    df["transactiondate"] = df["transactiondate"].astype(str)
    # add useful info
    df["transactiondate"] = "q" + df["transactiondate"] + " transactiondate"
    df = make_dummies(df, "transactiondate")
    df.drop(["transactiondate"], axis=1, inplace=True)
    return df


def wrangle_zillow(use_cache=True):
    """wrangle zillow data"""
    filename = "prepped_zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading prepped data from csv...")
        return pd.read_csv(filename)
    df = get_zillow_data()
    df = clear_parcel_id_duplicates(df)
    df = remove_outliers(df)
    df = get_single_units(df)
    df = handle_missing_values(df)
    df = drop_replace_nulls(df)
    df = handle_transactiondate(df)
    df.propertylandusedesc = "usecode" + df.propertylandusedesc
    df = make_dummies(df, "propertycountylandusecode")
    # convert year built to age
    df["age"] = 2017 - df["yearbuilt"]
    # below is not need becasue we end up with la county only after removing outliers and nulls
    # df = convert_fips(df)
    # drop collumns that are redundant
    columns_to_drop = [
        "id",
        "calculatedbathnbr",
        "calculatedbathnbr",
        "finishedsquarefeet12",
        "fullbathcnt",
        "roomcnt",
        "yearbuilt",
        "fips",
        "parcelid",
        "propertylandusetypeid",
        "regionidcounty",
        "unitcnt",
        "assessmentyear",
        "propertylandusedesc",
        "propertycountylandusecode",
        "heatingorsystemdesc",
        "propertyzoningdesc",
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)
    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


def split_train_validate_test(df, y_value, stratify=False):
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


def split_data(df, y_value="logerror"):
    """
    split data into train, validate, and test sets
    """
    # split data in to train, validate, and test sets
    train, validate, test = split_train_validate_test(df, y_value)
    # split train, validate, and test into x and y
    X_train, y_train = split_x_y(train, y_value)
    X_validate, y_validate = split_x_y(validate, y_value)
    X_test, y_test = split_x_y(test, y_value)
    return (
        train,
        validate,
        test,
        X_train,
        y_train,
        X_validate,
        y_validate,
        X_test,
        y_test,
    )


def scale_numeric_columns(X_train, X_validate, X_test, scaler=MinMaxScaler()):
    """scale numeric columns after fitting on training set"""
    X_train = X_train.copy()
    X_validate = X_validate.copy()
    X_test = X_test.copy()
    # choose numerical columns
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    # transform validation data
    X_validate[numerical_cols] = scaler.transform(X_validate[numerical_cols])
    # transform test data
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    return X_train, X_validate, X_test


def basic_info(df):
    """print some basic information about the dataframe"""
    print(df.info())
    print(df.describe())
    print("\n")
    print("null counts:")
    print(df.isnull().sum())
