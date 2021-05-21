# Main function

from .populate import *
from .process import *
from .clustering import *

STATES = ['12']
raw_data_db = '../data/raw_data.db'
proc_data_db = '../data/proc_data.db'


def main():
    
    populate_and_process(STATES, raw_data_db, proc_data_db)

    gdf = generate_cluster_shapefile(STATES, proc_data_db)
    


if __name__ == '__main__':
    main()