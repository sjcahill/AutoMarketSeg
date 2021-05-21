# Module contains functions that process and transform raw census
# data into a form more conducive for modelling.

import pickle
from populate import *

def process_tables(state, table_list, main_processing_dict, function_dict, input_db, output_db):
    '''
    Function that takes a list of table names corresponding to raw data ACS 5yr tables and 
    performs data processing to get in a suitable state for modelling.
    '''    

    for t in table_list:

        print(f'Processing table {t}')

        df = db_to_df('ST' + state + '_' + t, input_db)
        
        t_l = t.lower()
        df = function_dict[t_l](df, main_processing_dict[t_l])        
        t_name = 'ST' + state + '_' + t + 'p'
        
        df_to_db(df, output_db, t_name)

def parse_geo_id(df):
    '''
    Function that takes the GEO_ID column of census data and parses the value to create
    4 new columns containing components of State FIPS codes.
    '''

    dfc = df.copy()

    dfc['_fips'] = dfc.GEO_ID.apply(lambda x: x.split('US')[1] if len(x) > 3 else '_fips')
    dfc['STATEFP'] = dfc.GEO_ID.apply(lambda x: x.split('US')[1][:2] if len(x) > 3 else '_statefp')
    dfc['COUNTYFP'] = dfc.GEO_ID.apply(lambda x: x.split('US')[1][2:5] if len(x) > 3 else '_countyfp')
    dfc['TRACTCE'] = dfc.GEO_ID.apply(lambda x: x.split('US')[1][5:] if len(x) > 3 else '_tractce')
    
    dfc.drop(columns=['GEO_ID', 'NAME'], inplace=True)
    
    return dfc

def add_columns(df, target_column, cols_to_add):
    """
    Adds dataframe columns together.
    """
    
    df[target_column] = df[cols_to_add].sum(axis=1)
    return df

def fips_to_str(df):
    '''
    Ensures that the values in the fips columns are strings
    '''
    return df.astype({'_fips' : 'str'})

def main_dict_contents(main_dict):
    '''
    Un-nests the contents of the main processing dictionary for a better view on what
    the contents are.
    '''
    return [{k : [sk for sk in main_dict[k]]} for k in main_dict.keys()]

def process_table_generic(df, process_dict, divide=True):
    
    #tname = df.columns[2].split('_')[0]
    df = parse_geo_id(df)
    drop_cols = df.columns.difference(set(process_dict['columns']))
    df = df.drop(columns=drop_cols)
    df = df.rename(columns = process_dict['column_names'])
    df = df[process_dict['column_order']]
    df.iloc[:,:-4] = df.iloc[:,:-4].apply(pd.to_numeric)
    
    if divide:
        df = df[df.iloc[:,0] != 0]
        df.iloc[:,1:-4] = df.iloc[:,1:-4].div(df.iloc[:,0], axis=0)
    df.reset_index(inplace=True, drop=True) 
    
    #print(f'Successfully processed table {tname}')
    return df

def process_s0101(df, process_dict):
    '''
    Function to process the S1010 (Age/Sex) Subject Table. Ends up compressing age ranges into
    3 bins. 15-24, 25-49, 50+.
    
    Drops tracts that have a 0 total population
    '''
    
    df = parse_geo_id(df)
    drop_cols = df.columns.difference(set(process_dict['columns']))
    df = df.drop(columns=drop_cols).rename(columns = process_dict['column_names'])
    df.iloc[:, :-4] = df.iloc[:, :-4].apply(pd.to_numeric)
    
    add_columns(df, 'age_15_24', ['age_15_17', 'age_18_24'])
    add_columns(df, 'age_25_49', ['age_25_29', 'age_30_34', 'age_35_39', 'age_40_44', 'age_45_49'])
    add_columns(df, 'age_50+', ['age_50_54', 'age_60_64', 'age_65+'])
    
    df = df[process_dict['column_order']]
    
    df = df[df.total_pop != 0]   
    df.iloc[:,1:-4] = df.iloc[:,1:-4].div(df['total_pop'], axis=0)
    df.reset_index(inplace=True, drop=True)
    
    #print('Successfully processed table S0101')
    return df

def process_s1501(df, process_dict):
    """
    Function for processing the S1501 ACS 5yr subject table

    Args:
        df (DataFrame): raw census table as a dataframe
        process_dict (dict): Dictionary that provides mappings needed during process

    Returns:
        DataFrame: processed dataframe
    """
    
    f = parse_geo_id(df)
    df = df[process_dict['columns']]
    df = df.rename(columns = process_dict['column_names'])
    df.iloc[:,:-4] = df.iloc[:,:-4].apply(pd.to_numeric)
    
    df = add_columns(df, 'no_hs_diploma', ['no_hs_lt9', 'no_hs'])
    df = add_columns(df, 'hs_some_college', ['hs_sc', 'hs_ad'])
    df.drop(columns = ['no_hs_lt9', 'no_hs', 'hs_sc', 'hs_ad'], inplace=True)
    
    df = df[process_dict['column_order']]
    
    df = df[df.iloc[:,0] != 0]
    df.iloc[:,1:-4] = df.iloc[:,1:-4].div(df.iloc[:,0], axis=0)
    
    #print('Successfully processed table S1501')
    return df

def process_s1903(df, process_dict):
    """
    Function for processing the S1903 ACS 5yr subject table

    Args:
        df (DataFrame): raw census table as a dataframe
        process_dict (dict): Dictionary that provides mappings needed during process

    Returns:
        DataFrame: processed dataframe
    """
    
    def med_income_check(x):
        if isinstance(x, int):
            if x > 0:
                return False
    
    df = parse_geo_id(df)
    df = df[process_dict['columns']]
    df = df.rename(columns = process_dict['column_names'])
    df.iloc[:,:-4] = df.iloc[:,:-4].apply(pd.to_numeric)
    
    df = df[~(df.median_income == -666666666)].copy()    
    df = df[process_dict['column_order']]
    
    median = df.median_income.median()
    average = df.median_income.mean()
    
    assert df.median_income.apply(lambda x: med_income_check(x)).sum() == 0, 'Bad values in median income column for S1903'
        
    #print('Successfully processed table S1903')
    return df

def process_s2403(df, process_dict):
    """
    Function for processing the S2403 ACS 5yr subject table

    Args:
        df (DataFrame): raw census table as a dataframe
        process_dict (dict): Dictionary that provides mappings needed during process

    Returns:
        DataFrame: processed dataframe
    """

    df = parse_geo_id(df)
    df = df[process_dict['columns']]    
    
    df.iloc[:,:-4] = df.iloc[:,:-4].apply(pd.to_numeric)
    df = df[df.iloc[:,0] != 0]
    df.iloc[:,1:-4] = df.iloc[:,1:-4].div(df.iloc[:,0], axis=0)
    
    new_df = pd.DataFrame(columns = [
                                    'manufct_const',
                                    'sales',
                                    'util_logist',
                                    'fin_it_re',
                                    'sci_tech',
                                    'wastemgmt_admin',
                                    'edu_health_social',
                                    'hospitality',
                                    'public_admin',
                                    '_fips',
                                    'STATEFP',
                                    'COUNTYFP',
                                    'TRACTCE'
                                ]
                            )
    
    new_df['manufct_const'] = df.iloc[:,1:5].sum(axis=1)
    new_df['sales'] = df.iloc[:,5:7].sum(axis=1)
    new_df['util_logist'] = df.iloc[:,7]
    new_df['fin_it_re'] = df.iloc[:,8:10].sum(axis=1)
    new_df['sci_tech'] = df.iloc[:,10]
    new_df['wastemgmt_admin'] = df.iloc[:,11]
    new_df['edu_health_social'] = df.iloc[:,12]
    new_df['hospitality'] = df.iloc[:,13]
    new_df['public_admin'] = df.iloc[:,14]
    new_df['_fips'] = df['_fips']
    new_df['STATEFP'] = df['STATEFP']
    new_df['COUNTYFP'] = df['COUNTYFP']
    new_df['TRACTCE'] = df['TRACTCE']    
    
    #print('Successfully processed table S2403')
    return new_df

def combine_tables(table_list, db_path, wanted_columns, state):
    """
    Takes individually processed ACS subject table for a state and 
    merges them into a single table to get ready for modelling.

    Args:
        table_list (List[str]): List of table names we want to combine
        db_path (str): Location of database with processed census data
        wanted_columns (List[str]): List of columns that we want in our output
        state (str): String of the numeric state FIPS code

    Returns:
        DataFrame: The result of the combining of tables
    """
    
    p_tables = ['ST' + state + '_' + t + 'p' for t in table_list]
    
    for i in range(len(p_tables)):
        if i == 0:
            df = db_to_df(p_tables[i], db_path)
        else:
            df = df.merge(db_to_df(p_tables[i], db_path), on='_fips', )
            
    df = df[wanted_columns]
    df_to_db(df, db_path, 'ST' + state + '_' + 'combined_tables')
    print(f'Successfully processesed state {state}')
    
    return df

def make_model_ready(df, db_path, pop_threshold, state):
    """
    Takes a table of combined census data and does some processing to get it ready
    to model.

    Eliminates tracts with very low populations.

    Also stores the max values for each column so we can unscale the data later.

    Args:
        df (DataFrame): Combined table of census data
        db_path (str): Location of processed data database
        pop_threshold (int): Some threshold for eliminating low pop tracts from table
        state (str): String of numeric state FIPS code

    Returns:
        DataFrame: Returns 
    """
    
    pop_df = df[['total_pop', '_fips']]
    df_to_db(pop_df, db_path, 'ST' + state + '_' + 'population')
    
    df = df[df.total_pop > pop_threshold].copy()
    df.index=df._fips
    df.drop(columns=[
                '_fips',
                'total_pop',
                '_statefp',
                '_countyfp',
                '_tractce'
                ],
            inplace=True
            )
    
    df = df[['median_income'] + [col for col in df.columns if col != 'median_income']]
    
    scale_values = df.iloc[:,:].max()
    svdf = scale_values.to_frame('max')
    svdf['colnames'] = svdf.index
    df_to_db(svdf, db_path, 'ST' + state + '_' +'scale_values')    
        
    return df

def populate_and_process(States, raw_data_db, proc_data_db):
    """
    Function that combines the above functions to iteratively process our
    raw census data. Processing is simply selecting only relevant columns
    and some renaming for easier reading later on.
    

    Args:
        States (List[str]): List of states fips codes for wanted states.
        raw_data_db ([type]): location of database for raw census data.
        proc_data_db ([type]): location of database for processed census data.
    """

    # The census tables that we want to retrieve from the census API. All ACS 5-year Subject Tables.
    tables = [
            'S0101',
            'S1201',
            'S1501',
            'S1601',
            'S1903',
            'S2403',
            'S2501'
            ]

    # Imports a nested dictionary that helps with processing the tables
    with open('../data/main_processing_dict.pickle', 'rb') as file:
        main_processing_dict = pickle.load(file)

    # Imports a dictionary that maps tables to their processing functions
    with open('../data/function_dict.pickle', 'rb') as file:
        function_dict = pickle.load(file)
        
    # Imports a list of columns we want to keep after combining all the processed tables.
    with open('../data/combined_column_filter.pickle', 'rb') as file:
        wanted_columns = pickle.load(file)

    # Dictionary that converts fips code to 2 letter state code.
    with open('../data/fips_to_state.pickle', 'rb') as file:
        fips_to_state = pickle.load(file)


    print('populating database with processed data.')

    for state in States:

        census_to_database(state, tables, raw_data_db)   
        
        process_tables(state, tables, main_processing_dict, function_dict, raw_data_db, proc_data_db)
        
        combine_tables(tables, proc_data_db, wanted_columns, state)
        
      






        
