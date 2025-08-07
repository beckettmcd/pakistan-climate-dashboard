#!/usr/bin/env python3
"""
Test script to verify data loading functionality
"""

import pandas as pd
import json
import geopandas as gpd

def test_data_loading():
    """Test the data loading functions"""
    print("Testing data loading...")
    
    try:
        # Test climate data loading
        print("Loading climate data...")
        climate_df = pd.read_csv('data/Sample_District_Climate_Data.csv')
        print(f"✓ Climate data loaded successfully: {len(climate_df)} districts")
        print(f"  Columns: {list(climate_df.columns)}")
        
        # Test geographic data loading
        print("Loading geographic data...")
        with open('data/geoBoundaries-PAK-ADM2 (1).geojson', 'r') as f:
            geo_data = json.load(f)
        print(f"✓ Geographic data loaded successfully: {len(geo_data['features'])} features")
        
        # Test GeoDataFrame creation
        print("Creating GeoDataFrame...")
        gdf = gpd.GeoDataFrame.from_features(geo_data['features'])
        gdf = gdf.rename(columns={'shapeName': 'district'})
        print(f"✓ GeoDataFrame created successfully: {len(gdf)} districts")
        
        # Test data merging
        print("Merging climate and geographic data...")
        merged_df = gdf.merge(climate_df, on='district', how='left')
        print(f"✓ Data merged successfully: {len(merged_df)} districts")
        
        # Check for missing data
        missing_climate = merged_df[merged_df['avg_temperature_c'].isna()]
        if len(missing_climate) > 0:
            print(f"⚠️  Warning: {len(missing_climate)} districts missing climate data")
        else:
            print("✓ All districts have climate data")
        
        # Display sample statistics
        print("\nSample Statistics:")
        print(f"Temperature range: {climate_df['avg_temperature_c'].min():.1f}°C - {climate_df['avg_temperature_c'].max():.1f}°C")
        print(f"Rainfall range: {climate_df['avg_rainfall_mm'].min():.1f}mm - {climate_df['avg_rainfall_mm'].max():.1f}mm")
        print(f"Drought risk range: {climate_df['drought_risk_index'].min():.2f} - {climate_df['drought_risk_index'].max():.2f}")
        print(f"Flood risk range: {climate_df['flood_risk_index'].min():.2f} - {climate_df['flood_risk_index'].max():.2f}")
        
        print("\n✅ All tests passed! Data loading is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error during data loading: {str(e)}")
        return False

if __name__ == "__main__":
    test_data_loading()
