from env import get_db_url
import pandas as pd
import os


def get_zillow_data(use_cache=True):
    """pull from SQL unless zillow.csv exists"""
    filename = "zillow.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("zillow")
    query = """
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, fips
    FROM properties_2017
    JOIN propertylandusetype using (propertylandusetypeid) 
    JOIN predictions_2017 using (parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
    AND transactiondate like "2017%%";
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df


# this is used to import different data from sql if needed in the future
def get_zillow_data2(use_cache=True):
    """Same as above but pulls in more squarefoot data, pull from SQL unless zillow.csv exists"""
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


def get_iris_data(use_cache=True):
    """pull from SQL unless iris.csv exists"""
    filename = "iris.csv"
    if os.path.isfile(filename) and use_cache:
        print("Reading from csv...")
        return pd.read_csv(filename)

    print("reading from sql...")
    url = get_db_url("iris_db")
    query = """
    SELECT *
    FROM measurements
    JOIN species USING(species_id)
    """
    df = pd.read_sql(query, url)

    print("Saving to csv in local directory...")
    df.to_csv(filename, index=False)
    return df
