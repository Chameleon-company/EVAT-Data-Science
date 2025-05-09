import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class PowerBIDataPrep:
    def __init__(self, input_file=None):
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        if input_file is None:
            data_dir = os.path.join(self.current_dir, 'data')
            files = [f for f in os.listdir(data_dir) if f.startswith('validated_data_')]
            input_file = os.path.join(data_dir, sorted(files)[-1])
        
        self.df = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}")
        
    def prepare_operational_metrics(self):
        """Prepare operational status metrics with better unknown handling"""
        # Convert operational status to numeric
        self.df['operational_score'] = self.df['is_operational'].map({
            'Yes': 1.0,
            'No': 0.0,
            'Unknown': 0.5  # Assign middle score for unknown
        }).fillna(0.5)
        
        # Calculate charging capacity
        self.df['charging_capacity'] = self.df['charging_points'] * self.df['power_output'].fillna(0)
        
        # Create accessibility score (0-1) with unknown handling
        accessibility_factors = ['pay_at_location', 'membership_required', 'access_key_required']
        for factor in accessibility_factors:
            self.df[f'{factor}_score'] = self.df[factor].map({
                'Yes': 0.0,
                'No': 1.0,
                'Unknown': 0.5  # Middle score for unknown
            }).fillna(0.5)
        
        self.df['accessibility_score'] = (
            self.df['pay_at_location_score'] + 
            self.df['membership_required_score'] + 
            self.df['access_key_required_score']
        ) / 3
        
        return self
    
    def calculate_station_density(self):
        """Calculate station density for heatmap"""
        # Create grid cells for density calculation
        lat_bins = np.linspace(self.df['latitude'].min(), self.df['latitude'].max(), 50)
        lon_bins = np.linspace(self.df['longitude'].min(), self.df['longitude'].max(), 50)
        
        # Calculate station density with observed=True
        self.df['station_density'] = self.df.groupby(
            [pd.cut(self.df['latitude'], lat_bins),
             pd.cut(self.df['longitude'], lon_bins)],
            observed=True
        )['charging_points'].transform('count')
        
        # Normalize density
        self.df['station_density_normalized'] = (
            self.df['station_density'] - self.df['station_density'].min()
        ) / (self.df['station_density'].max() - self.df['station_density'].min())
        
        return self
    
    def create_power_bi_dataset(self):
        """Prepare final dataset for Power BI"""
        # Select and rename columns for Power BI
        powerbi_df = self.df[[
            'latitude', 'longitude',
            'charging_points', 'power_output',
            'operational_score', 'accessibility_score',
            'station_density', 'station_density_normalized',
            'operator', 'connection_type', 'current_type',
            'charging_capacity'
        ]].copy()
        
        # Clean up and standardize values
        powerbi_df['operator'] = powerbi_df['operator'].fillna('Unknown')
        powerbi_df['connection_type'] = powerbi_df['connection_type'].fillna('Unknown')
        powerbi_df['current_type'] = powerbi_df['current_type'].fillna('Unknown')
        powerbi_df['power_output'] = powerbi_df['power_output'].fillna(0)
        
        # Calculate overall station score with weighted components
        powerbi_df['station_score'] = (
            (powerbi_df['operational_score'] * 0.3) +
            (powerbi_df['accessibility_score'] * 0.2) +
            (powerbi_df['charging_capacity'] / powerbi_df['charging_capacity'].max() * 0.3) +
            (powerbi_df['station_density_normalized'] * 0.2)
        )
        
        return powerbi_df
    
    def save_for_powerbi(self):
        """Save prepared data for Power BI"""
        # Prepare the data
        self.prepare_operational_metrics()
        self.calculate_station_density()
        powerbi_df = self.create_power_bi_dataset()
        
        # Save file
        output_path = os.path.join(
            self.current_dir, 
            'data', 
            f'powerbi_data_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        )
        powerbi_df.to_csv(output_path, index=False)
        
        # Print summary
        print("\nData Summary for Power BI:")
        print(f"Total stations: {len(powerbi_df)}")
        print("\nValue ranges:")
        numeric_cols = powerbi_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            print(f"{col}:")
            print(f"  Min: {powerbi_df[col].min():.2f}")
            print(f"  Max: {powerbi_df[col].max():.2f}")
            print(f"  Mean: {powerbi_df[col].mean():.2f}")
        
        # Print unique values for categorical columns
        categorical_cols = ['operator', 'connection_type', 'current_type']
        print("\nUnique values in categorical columns:")
        for col in categorical_cols:
            print(f"\n{col}:")
            print(powerbi_df[col].value_counts().head())
        
        print(f"\nData saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    prep = PowerBIDataPrep()
    output_file = prep.save_for_powerbi()