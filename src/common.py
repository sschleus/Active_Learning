
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT_DATASETS = ROOT / "data" / "init_datasets"
PROCESSED_DATASETS = ROOT / "data" / "processed_datasets"
EXPERIMENTS = ROOT / "data" / "experiments"
HYBRID_ANALYSIS = ROOT / "data" / "hybrid_analysis"
CONNECTIVITY_PATH = INIT_DATASETS / "osm_connectivity_features.parquet"
DISTCITY_PATH = INIT_DATASETS / "distcity.csv"
SPATIAL_DENSITY_PATH = INIT_DATASETS / "osm_centroid_features.parquet"
D_PATH = INIT_DATASETS / "spw_osm_training_set.parquet"
L_OSM_ID_PATH = INIT_DATASETS / "sensors_id.csv"
L_PATH = PROCESSED_DATASETS / "L.joblib"
U_PATH = PROCESSED_DATASETS / "U.joblib"

HIGHWAY_TYPES = ['highway_secondary', 'highway_primary',
       'highway_motorway_link', 'highway_trunk', 'highway_motorway', 'highway_tertiary', 'highway_trunk_link']
SPEED_TYPES = ['maxspeed']
LANES_TYPES = ['lanes']

BASE = ['lon', 'lat', 'highway_motorway', 'highway_motorway_link', 'highway_primary', 
                  'highway_secondary', 'highway_tertiary', 'highway_trunk', 'highway_trunk_link', 
                   'maxspeed', 
                   'lanes', 'target'] 
DISTANCE = ['dist_bruxelles','dist_liege', 'dist_namur', 'dist_charleroi', 'dist_mons','dist_la_louviere',
            'dist_tournai','dist_seraing','dist_verviers','dist_herstal', 'dist_mouscron']
SPATIAL_DENSITY  = ["knn_distance","local_density"]
CONNECTIVITY = ["neighbor_maxspeed_mean", "num_neighbors"]
ORDER = (BASE[:-1] + DISTANCE + CONNECTIVITY + SPATIAL_DENSITY + ["target"])
HIGHWAY_MEDIUM = ['highway_secondary', 'highway_primary','highway_motorway_link', 'highway_tertiary', 'highway_trunk_link']

NO_MODEL = 0
SINGLE = 1
MULTI = 2
TEST = 3
UNIFORM_RANDOM_SAMPLING = 4
STRATIFIED_RANDOM_SAMPLING = 5
RANDOM_SAMPLING_HIGH_TRAFFIC_ONLY = 6
UNCERTAINTY_ONLY = 7
FURTHEST_SAMPLING = 8
HYBRID = 9