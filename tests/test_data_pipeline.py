import pandas as pd
import numpy as np
import os
import pytest
from src.data_pipeline import load_and_clean_data

@pytest.fixture
def mock_data_file(tmp_path):
    df = pd.DataFrame({
        'trip_uuid': ['trip_1', 'trip_2', 'trip_3', 'trip_4'],
        'source_center': ['A', 'B', 'C', 'D'],
        'source_name': ['Alpha', np.nan, 'Charlie', 'Delta'],
        'destination_center': ['B', 'C', 'D', 'E'],
        'destination_name': ['Beta', 'Charlie', np.nan, 'Echo'],
        'segment_actual_time': [100.0, -50.0, 120.0, 45.0],  # trip_2 has negative time
        'segment_osrm_time': [80.0, 60.0, 0.0, 40.0],      # trip_3 has zero OSRM time
        'segment_osrm_distance': [100, 80, 150, 50],
        'trip_creation_time': ['2023-01-01 08:30:00', '2023-01-01 14:00:00', '2023-01-01 19:45:00', '2023-01-01 23:10:00'],
        'route_type': ['FTL', 'Carting', 'FTL', 'Carting']
    })
    file_path = tmp_path / "mock_delivery_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def test_negative_time_filtering(mock_data_file):
    df_clean = load_and_clean_data(mock_data_file)
    # The row with negative time (trip_2) should be dropped
    assert len(df_clean) == 3
    assert not (df_clean['trip_uuid'] == 'trip_2').any()

def test_missing_names_handling(mock_data_file):
    df_clean = load_and_clean_data(mock_data_file)
    # Missing source name in B should be Unknown_Source
    assert df_clean.loc[df_clean['trip_uuid'] == 'trip_1', 'source_name'].iloc[0] == 'Alpha'
    # Actually trip 2 is dropped, so we can't test its missing source name
    # But trip 3 has a missing destination name
    assert df_clean.loc[df_clean['trip_uuid'] == 'trip_3', 'destination_name'].iloc[0] == 'Unknown_Destination'

def test_zero_osrm_handling(mock_data_file):
    df_clean = load_and_clean_data(mock_data_file)
    # trip 3 had zero OSRM time, it should be changed to 1.0
    val = df_clean.loc[df_clean['trip_uuid'] == 'trip_3', 'segment_osrm_time'].iloc[0]
    assert val == 1.0

def test_segment_factor_calculation(mock_data_file):
    df_clean = load_and_clean_data(mock_data_file)
    # trip 1 factor: 100 / 80 = 1.25
    factor_1 = df_clean.loc[df_clean['trip_uuid'] == 'trip_1', 'segment_factor'].iloc[0]
    assert factor_1 == 1.25

def test_time_of_day_binning(mock_data_file):
    df_clean = load_and_clean_data(mock_data_file)
    # 08:30 -> Morning
    assert df_clean.loc[df_clean['trip_uuid'] == 'trip_1', 'time_of_day'].iloc[0] == 'Morning'
    # 19:45 -> Evening
    assert df_clean.loc[df_clean['trip_uuid'] == 'trip_3', 'time_of_day'].iloc[0] == 'Evening'
    # 23:10 -> Night
    assert df_clean.loc[df_clean['trip_uuid'] == 'trip_4', 'time_of_day'].iloc[0] == 'Night'
