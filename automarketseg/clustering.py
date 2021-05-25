import sqlite3 as db
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
from sklearn.cluster import KMeans
from process import *
from populate import get_tract_shapefile

# For visualizations
import matplotlib
import geoviews as gv
import hvplot.pandas
gv.extension('matplotlib', 'bokeh')
gv.output(backend='bokeh')

def scale_data(df):
    """
    Uses the MinMaxScaler function from sklearn to scale all data to [0,1]

    Args:
        df (DataFrame): dataframe of combined census data

    Returns:
        DataFrame: dataframe scaled to [0,1]
    """       
    scaler=MinMaxScaler()
    
    return pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

def pca_outlier(df):
    """
    Uses the PCA outlier detection method implemented in the pyod package.
    Which is based on sklearns method.

    Args:
        df (DataFrame): dataframe of combined census data

    Returns:
        fitted estimator: fitted estimator object from PCA of dataframe
    """
    clf = PCA()
    
    return clf.fit(df)

def iforest_outlier(df):
    """
      Uses the PCA outlier detection method implemented in the pyod package.
      Which is based on sklearns method.

      Args:
        df (DataFrame): dataframe of combined census data

      Returns:
        fitted estimator: fitted estimator object from PCA of dataframe
    """    
    clf = IForest(n_estimators=250, behaviour='old')
    
    return clf.fit(df)

def create_ds_df(df, clf, colname):
    """
        Returns a dataframe with an additional column that representes
        decision scores from a fitted estimator, the result of 
        training an Outlier Detection Method on the dataframe.

    Args:
        df (DataFrame): Data the fitted estimator was generated from
        clf (fitted estimator): fitted estimator object result from outlier detection
        colname (str): Name to call the column.

    Returns:
        DataFrame: dataframe with decision scores from outlier detection appended.
    """

    ds = df.copy()
    ds[colname] = clf.decision_scores_
    ds[colname] = (ds[colname] - ds[colname].min())/(ds[colname].max()-ds[colname].min())
    
    return ds

def sort_ds_scores(df, sort_cols, db, num_rows=None):
    """
    Sorts dataframes by their decision scores

    Args:
        df (DataFrame): df to sort
        sort_cols (List or str): column to sort on
        db (str): location of database with population tables
        num_rows (int): Number of rows to return. Defaults to None.

    Returns:
        [type]: [description]
    """
    
    if num_rows is None:
        return df.sort_values(by=sort_cols, ascending=False).merge(db_to_df('population', db), on='_fips')
        
    return df.sort_values(by=sort_cols, ascending=False).head(num_rows).merge(db_to_df('population', db), on='_fips')


def generate_elbow_diagram(df, num_clusters, num_iters=5):
    """
    Generates an elbow diagram which shows how increasing the number of clusters
    reduces the Sum of Square Distances (Euclidean)

    Args:
        df (DataFrame): dataframe to be analyzed
        num_clusters (int): Upper threshold for cluster number
        num_iters (int, optional): How many cluster analyses to run for each cluster number
                                   Defaults to 5.
    """
        
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    
    distortions = []
    for i in range(2, num_clusters + 1):
        inertia = 0
        for n in range(num_iters):
            km = KMeans(n_clusters = i)
            km.fit(df)
            inertia += km.inertia_
        distortions.append(inertia/num_iters)
        
    plt.plot(range(2, num_clusters+1), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

def cluster_aggregate_table(df, scale_values):
    """TODO: write some stuff.

    Args:
        df ([type]): [description]
        scale_values ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    sv = scale_values['max'].values 
    cols = [col for col in df.columns if col not in ['total_pop', 'cluster']]    
    
    df = df.groupby('cluster')[cols].mean().multiply(sv, axis=1).round(decimals=3).merge(df.groupby('cluster')['total_pop'].sum(), on='cluster')
    
    col_order = ['total_pop', 'median_income']  + [col for col in df.columns if col not in ['total_pop', 'median_income']]
    df = df[col_order]
    
    return df

def parse_down_shapefile(gdf):
    """
    Takes a shapfile - generally a cenus data shapefile - and returns 
    only relevant columns with a renaming as well.

    Args:
        gdf (GeoDataFrame): geodataframe generated from shapefile using `geopandas`

    Returns:
        GeoDataFrame: Returns the modified GeoDataFrame.
    """

    gdf = gdf[['GEOID', 'geometry']].copy()
    gdf.rename(columns={'GEOID' : '_fips'}, inplace=True)
    gdf = gdf.astype({'_fips' : str})
    gdf.index = gdf._fips
    gdf.drop(columns='_fips', inplace=True)
    
    return gdf

def crude_outlier_detection(df, threshold, state):
    """
    This function uses the results of PCA and Iforest outlier detection
    methods on a dataset and, using some largely arbitrary threshold,
    removes datapoints that lie above the score thresholds as outliers.

    Args:
        df (DataFrame): df to perform outlier detection on
        threshold (float): score threshold to determine outliers
        state (str): State fips code?

    Returns:
        [type]: [description]
    """

    clf1 = pca_outlier(df)
    ds = create_ds_df(df, clf1, 'ds_pca')
    clf2 = iforest_outlier(df)
    ds = create_ds_df(ds, clf2, 'ds_iforest')
    before = ds.shape[0]
    ds = ds[(ds.ds_pca < threshold) & (ds.ds_iforest < threshold)]
    after = ds.shape[0]
    print(f'Dropped {before - after} tracts in state {state} due to outlier detection. PCA/Isolation Forest.')
    
    #Drop the added outlier scores as they are no longer necessary.
    result_df = ds.drop(columns=['ds_pca', 'ds_iforest'])
    
    return result_df

def perform_cluster_analysis(df, k):
    """
    Returns the result of a cluster analysis using k clusters

    Args:
        df (DataFrame): 
        k (int): number of clusters
    Returns:
        DataFrame: dataframe with corresponding cluster labels attached
    """

    print('Performing cluster analysis.')
    
    clf_km = KMeans(n_clusters=k)
    clf_km.fit(df)
    df['cluster'] = clf_km.labels_
    
    return df

def generate_regional_cluster_shapefile(states, engine, cluster_df, name):
    """
    Generates a shapefile that is the result of a regional cluster analysis.
    Takes the necessary shapefiles from a Postgre database and then joins
    the results of the cluster analysis onto them.

    Example. Mid-Atlantic: MD, DC, and VA

    Args:
        states (List[str]): list of states to use in analysis
        engine (sqlalchemy): engine to postGRE database
        cluster_df (DataFrame): result of regional cluster analysis
        name (str): Name of the resulting file
    """
    
    main_shape = gpd.GeoDataFrame()
    
    for state in states:
        with engine.connect() as connection:
            shapefile = shapefile = gpd.read_postgis(state, connection, geom_col='geometry');
            shapefile = parse_down_shapefile(shapefile)
            main_shape = main_shape.append(shapefile)   
    
    cluster_df = main_shape.merge(cluster_df, on='_fips')
    
    with engine.connect() as connection:
        cluster_df.to_postgis('cluster' + '_' + name, connection, if_exists='replace')
    
    print(f'Successfully processed region {name}')


def generate_cluster_shapefile(States, proc_data_db, pop_threshold=400):

    print("In generate cluster shapefiles.")

    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    for state in States:

        
        data = db_to_df('ST' + state + '_' + 'combined_tables', proc_data_db)
        data = make_model_ready(data, proc_data_db, pop_threshold, state)
        scale_values = db_to_df('ST' + state + '_' + 'scale_values', proc_data_db)
        pop = db_to_df('ST' + state + '_' + 'population', proc_data_db)
        
        data_s = scale_data(data)

        data_s = crude_outlier_detection(data_s, .7, state)
        result = perform_cluster_analysis(data_s, 6)
        result = result.reset_index().merge(pop, how='left').set_index(result.index.names)
        agg_table = cluster_aggregate_table(result, scale_values)

        df_to_db(agg_table, proc_data_db, table_name = 'ST' + state + '_' + 'agg_table')
        
        gdf = get_tract_shapefile(state)
        gdf = parse_down_shapefile(gdf)
        result = gdf.merge(result, on='_fips')

    print(f'Successfully processed state {state}')

    return result

def generate_cmap(df):
    num_clusters = len(df.cluster.unique())
    cmap = matplotlib.colors.ListedColormap([matplotlib.cm.get_cmap("tab10").colors[i] for i in list(range(num_clusters))])
    return cmap   
    

def generate_interactive_map(df):
    
    cmap = generate_cmap(df)    
    return df.hvplot(c='cluster', cmap=cmap, line_width=0.1, alpha=0.7,  geo=True, tiles='CartoLight',  xaxis=False, yaxis=False, frame_height=900, frame_width=1200, colorbar=True)



