import pandas as pd
import os
from env import get_db_url

# get_zillow_data while maintaining my original query from previous project
def get_zillow_data(use_cache=True):
    """pull from SQL unless zillow.csv exists"""
    filename = "zillow2.csv"
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


def wrangle_zillow(use_cache=True):
    """wrangle zillow data"""
    filename = "prepped_zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading prepped data from csv...")
        return pd.read_csv(filename)
    df = get_zillow_data()
    # drop collumns that are redundant
    columns_to_drop = [
        "id",
        "calculatedbathnbr",
        "calculatedbathnbr",
        "finishedsquarefeet12",
        "fullbathcnt",
    ]
    df.drop(columns_to_drop, axis=1, inplace=True)
    df = clear_parcel_id_duplicates(df)
    df = get_single_units(df)
    df = handle_missing_values(df)
    df = drop_replace_nulls(df)
    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df
