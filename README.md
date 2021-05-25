<h1> Automatic Market Segmentation </h1>

<h2> Overview </h2>

The aim of this project is to provide a way to automate market segmentation
at the census tract level for any state in the US. For the purposes of this repo, 
I will be focusing on Florida, but this approach applies to any state or territory
that has data provided by [US Census ACS 5 year survey Subject Tables](https://data.census.gov/cedsci/all?d=ACS%205-Year%20Estimates%20Subject%20Tables).

This is mostly educational but touches on many important aspects of data analysis from 
end to end.

<br/>

<h3> General Steps </h3>

<br/>

1. Retrieve data from census API using an https request.
2. Perform necessary processing to get data in the form we want.
    - Select what columns we want from each table
    - Perform some necessary division using the baseline population of each subject table (they sometimes differ)
3. Perform outlier detection
4. Perform kmeans clustering to identify key demographic groups within a state.
5. Output report detailing the findings.
6. Output map to show clusters geographically.

<br/>

<h3> Data Storage </h3>

<br/>

I used SQLite dabatases via the sqlite3 package to store data. SQLite databases are stored
locally in a `.db` file and make for very easy storage and retrieval without having to create
a messy file structure.

**Note :** Since I was performing this analysis across multiple states and wanted to display my results on a map, 
I was using census shape files. These were stored in a Postgre database since it allowed me to use 
the very nice **postGIS** extension that enables `geography` datatype columns.

<br/>

Below are the US Census ACS 5 year subject table codes corresponding to tables I chose to use
for this analysis. There are many other tables that might be worthwhile to explore, but I found
these to be useful for my purposes.

{ 'S0101' : 'Age and Sex' ,  
  'S1201' : 'Marital Status' ,  
  'S1501' : 'Educational Attainment',  
  'S1601' : 'Language Spoken at Home',  
  'S1903' : 'Median Income Last 12 Months',  
  'S2403' : 'Industry by Sex for the Civilian Employed Population 16+',  
  'S2501' : 'Occupancy Characteristics'}


I also chose to download, process, and store the metadata tables that were associated with the regular data
to keep track of what columns were what and I even attempted to use them as string values for columns, but
that was more of a headache than it was worth.

--------------

Because I wanted to generate maps as part of the visual reporting to show the geographic distribution of clusters, 
I relied on census data shapefiles with polygon geometries for census tracts.

Shapefiles for 2019 are available at [https://www2.census.gov/geo/tiger/TIGER2019/TRACT/]['https://www2.census.gov/geo/tiger/TIGER2019/TRACT/']

An example of how to use python to download and extract a zipped shapefile directly into a 
geoDataFrame:

```python
import requests, io, geopandas as gpd
zip_file_url = 'https://www2.census.gov/geo/tiger/TIGER2019/TRACT/tl_2019_12_tract.zip'
r = requests.get(zip_file_url)
gdf = gpd.read_file(io.BytesIO(r.content))
```

Here is a shell command I found via Stack Overflow that will batch download all of the shapefiles for every
state and territory at once - it can probably be tweaked for a cleaner result:

`wget --execute="robots = off" --mirror --convert-links --no-parent --wait=5 'https://www2.census.gov/geo/tiger/TIGER2019/TRACT/'`

-----------

<h3> Final Notes </h3>

An html file was generated featuring the results of an analysis. It used an underlying jupyter notebook located in the `automarketseg`
directory.

The pipenv generated `Pipfile` and `requirements.txt` are sufficient to do everything EXCEPT generate the map. For some reason
`pipenv` was not playing nicely with the `geoviews` package.

Instead I used the a `conda` environment that solve the dependency conflicts for me. I generally use `conda` for this reason.

So, I have included the `environment.yml` for that in case you are interested.

Finally, if you come across this and see some egregious programming/analytics errors/bad practices, feel free
to let me know. I welcome any constructive criticism.
