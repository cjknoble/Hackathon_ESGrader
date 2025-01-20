import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from geopy.geocoders import Nominatim


# Function to process EJScreen data
def process_ejscreen(town_name, shapefile_path, cejst_shapefile_path):
    # Geocode the town name
    geolocator = Nominatim(user_agent="ejscreen_analysis")
    location = geolocator.geocode(town_name)
    if location is None:
        raise ValueError(f"Unable to geocode the town name: {town_name}")

    center_point = Point(location.longitude, location.latitude)

    # Read EJScreen shapefile
    ejscreen_data = gpd.read_file(shapefile_path)

    # Create a 5-mile buffer around the input location
    buffer = gpd.GeoSeries([center_point]).set_crs(epsg=4326).to_crs(epsg=3857).buffer(8046.72).to_crs(epsg=4326)

    # Clip EJScreen data to the buffer
    ejscreen_within_buffer = ejscreen_data[ejscreen_data.geometry.within(buffer.iloc[0])]

    # Calculate averages for selected columns
    selected_columns = [
        'P_PEOPCOLO', 'P_LOWINCPC', 'P_UNEMPPCT', 'P_DISABILI', 'P_LINGISOP',
        'P_LESSHSPC', 'P_UNDER5PC', 'P_OVER64PC', 'P_LIFEEXPP', 'P_PM25', 'P_OZONE',
        'P_DSLPM', 'P_RSEI_AIR', 'P_PTRAF', 'P_LDPNT', 'P_PNPL', 'P_PRMP', 'P_PTSDF',
        'P_UST', 'P_PWDIS', 'P_NO2', 'P_DWATER'
    ]

    averages = ejscreen_within_buffer[selected_columns].mean().reset_index()
    averages.columns = ['Column', 'Average Value']

    # Read CEJST shapefile
    cejst_data = gpd.read_file(cejst_shapefile_path)

    # Check if the buffer intersects with any polygons in CEJST
    intersects = cejst_data.geometry.intersects(buffer.iloc[0]).any()

    # Print the eligibility message
    cejst_check = 1 if intersects else 0

    return averages, cejst_check


# Function to calculate EJ indexes
def ej_indexes(results_df):
    """Create a DataFrame with demographic and environmental justice scores."""
    # Get the top social and environmental variables
    top_social = top_social_variables(results_df)
    top_environmental = top_environmental_variables(results_df)

    # Create the DataFrame with the maximum values
    location_data = pd.DataFrame({
        'demographic_index': [top_social['Average Value'].max()],
        'environmental_justice_score': [top_environmental['Average Value'].max()]
    })
    return location_data


# Function to extract top 3 Social variables
def top_social_variables(results_df):
    social_columns = {
        'P_PEOPCOLO': '% People of Color',
        'P_LOWINCPC': '% Low Income',
        'P_UNEMPPCT': '% Unemployed',
        'P_DISABILI': '% Persons with Disabilities',
        'P_LINGISOP': '% Limited English Speaking',
        'P_LESSHSPC': '% Less than High School Education',
        'P_UNDER5PC': '% Under Age 5',
        'P_OVER64PC': '% Over Age 64',
        'P_LIFEEXPP': 'Low Life Expectancy'
    }
    filtered_df = results_df[results_df['Column'].isin(social_columns.keys())]
    top_3 = filtered_df.nlargest(3, 'Average Value')
    top_3['Column'] = top_3['Column'].map(social_columns)
    return top_3


# Function to extract top 3 Environmental variables
def top_environmental_variables(results_df):
    environmental_columns = {
        'P_PM25': 'Particulate Matter 2.5',
        'P_OZONE': 'Ozone',
        'P_DSLPM': 'Diesel Particulate Matter',
        'P_RSEI_AIR': 'Toxic Releases to Air',
        'P_PTRAF': 'Traffic Proximity',
        'P_LDPNT': 'Lead Paint',
        'P_PNPL': 'Superfund Proximity',
        'P_PRMP': 'RMP Facility Proximity',
        'P_PTSDF': 'Hazardous Waste Proximity',
        'P_UST': 'Underground Storage Tanks',
        'P_PWDIS': 'Wastewater Discharge',
        'P_NO2': 'Nitrogen Dioxide (NO2)',
        'P_DWATER': 'Drinking Water Non-Compliance'
    }
    filtered_df = results_df[results_df['Column'].isin(environmental_columns.keys())]
    top_3 = filtered_df.nlargest(3, 'Average Value')
    top_3['Column'] = top_3['Column'].map(environmental_columns)
    return top_3
