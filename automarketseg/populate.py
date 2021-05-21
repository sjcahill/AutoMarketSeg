# Module contains the functions needed to request census API data
# and store it as tables in SQLite databases.

import requests
import pandas as pd
import sqlite3 as db


def census_to_database(state_fips, table_list, db_path):
    """Makes two API requests to census API and stores the results
    in an SQLite database

    Args:
        state_fips (str): state FIPS code
        table_list (List[str]): list of strings corr to ACS 5 yrSubject Tables
        db_path (str): path to location of database
        api_key (str): API key for accesssing census data. Optional for small volume.
    """    


    print(f'Populating raw census database for state fips code {state_fips}.')


                
    for t in table_list:

        print(f'Ingesting census data for table {t}')

        
        df = census_api_table_query(state_fips, t)
        mdf = census_api_metadata_query(t) #metadata same for all states
        df_to_db(df, db_path, 'ST' + state_fips + '_' + t)
        df_to_db(mdf, db_path, 'ST' + state_fips + '_' + t + '_metadata')
        
        
    
    print(f'Finished populating raw census database for state fips code {state_fips}')

def census_api_table_query(state_fips, table_name):

    assert isinstance(state_fips, str) and isinstance(table_name, str)
    
    query = f'https://api.census.gov/data/2019/acs/acs5/subject?get=group({table_name})&for=tract:*&in=state:{state_fips}&key='
   
    r = requests.get(query)
    df = json_to_df(r.json())

    try:
        df = df.drop(columns = ['state', 'county', 'tract'])
    except KeyError:
        return df
    return df
    
def census_api_metadata_query(table_name):

    assert isinstance(table_name, str)

    query = f'https://api.census.gov/data/2019/acs/acs5/subject/groups/{table_name}.json'
    
    r = requests.get(query)
    variables = r.json()['variables']
    df = var_dict_to_df(variables).sort_values(by='variables', ignore_index=True)

    return df

def json_to_df(requests_json):
    """
    Takes the json attribute from the query object received from API request
    and upacks into a DataFrame.

    Args:
        requets_json (dict): JSON dictionary returned from https request
        

    Returns:
        DataFame: Dataframe of subject table with filtered columns
    """    
    
    df = pd.DataFrame(data=requests_json[1:], columns=requests_json[0])

    filter_list = ['MA', 'EA']
    new_cols = [c for c in df.columns if c[-2:] not in filter_list]

    return df[new_cols]

def df_to_db(df, db_path, table_name=None):
    """
    Takes a DataFrame and inputs it as a table in an SQLite db


    Args:
        df (DataFrame): dataframe to input as table into db
        db_path (string): path to db
        table_name (str): Name of the table. Defaults to None.
    """    

    if (table_name is None) & (len(df.columns) > 2):
        table_name = df.columns[2][:5]
    elif (table_name is None) & (len(df.columns) == 2):
        table_name = df.iloc[0,0][:5] + '_metadata'
    
    
    with db.connect(db_path) as conn:
        c = conn.cursor()
        c.execute(f'DROP TABLE IF EXISTS {table_name}')
        df.to_sql(name=table_name, con=conn, index=False)        
        

def db_to_df(table_name, db_path):
    """
    Returns a table in SQLite db as a dataframe.


    Returns:
        DataFrame: df of census data.
    """    
    
    with db.connect(db_path) as conn:
        query = f'SELECT * FROM {table_name}'        
        df =  pd.read_sql(query, conn)
    
    return df

def var_dict_to_df(variables):
    """
    Takes the a dictionary of variables, selects the values we want, 
    and then returns an appropriate dataframe.

    Args:
        Dict[str]: dict of column names and there associated descriptions.
    Returns:
        DataFrame: dataframe of columns with descriptions as values
    """

    list1 = [(key, value['label']) for key, value in variables.items()]
    
    filter_list = ['MA', 'EA']
    
    dict1 = dict([x for x in list1 if x[0][-2:] not in filter_list])
    
    return pd.DataFrame({'variables' : dict1.keys(), 'description' : dict1.values()})

def get_tract_shapefile(state_fips):

    import io, geopandas as gpd    

    zip_file_url = f'https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_{state_fips}_tract.zip'
    r = requests.get(zip_file_url)
    gdf = gpd.read_file(io.BytesIO(r.content))

    return gdf

def win_to_lin(path: str) -> str:
    '''
    Tranlates the path to my windows C:\ drive to the 
    c mnt in my wsl ubuntu path
    
    Parameters:
        windows path string
    
    Returns:
        corresponding linux path string
    '''    
    path = path.replace('\\', '/')
    path = '/mnt/c' +  r'{}'.format(path[2:])
    return path



    






        


