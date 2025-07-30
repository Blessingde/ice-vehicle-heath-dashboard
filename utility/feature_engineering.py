import pandas as pd
import numpy as np

# --- Define Feature Sets ---
common_env_usage_features = [
    'Ambient_Temperature', 'Ambient_Humidity', 'Load_Weight',
    'Driving_Speed', 'Idle_Time', 'Route_Roughness', 'Distance_Traveled'
]

engine_specific_features = [
    'Engine_RPM', 'Engine_Oil_Pressure', 'Engine_Coolant_Temp',
    'Engine_Vibration', 'Exhaust_Gas_Temp', 'Check_Engine_Light_On',
    'Oil_Life_Remaining_Pct'
]
brake_specific_features = [
    'Brake_Pad_Wear_Front_mm', 'Brake_Pad_Wear_Rear_mm',
    'Brake_Fluid_Level_Pct', 'Brake_Temperature_Avg_C'
]
tire_specific_features = [
    'Tire_Pressure_FL_PSI', 'Tire_Pressure_FR_PSI',
    'Tire_Pressure_RL_PSI', 'Tire_Pressure_RR_PSI',
    'Tire_Temperature_Avg_C', 'Suspension_Load'
]
vehicle_metadata_features = ['Vehicle_Make', 'Vehicle_Model']

# The target lists are included here for completeness and to match the signature from train_models.py
engine_targets = [
    'Engine_RUL', 'Engine_TTF', 'Engine_Failure_Probability', 'Engine_Health_Score'
]
brake_targets = [
    'Brake_RUL', 'Brake_TTF', 'Brake_Failure_Probability', 'Brake_Health_Score'
]
tire_targets = [
    'Tire_RUL', 'Tire_TTF', 'Tire_Failure_Probability', 'Tire_Health_Score'
]


def prepare_features_for_component(data_df, component_specific_features_list, component_targets_list):
    """
    Applies time-series feature engineering and identifies final feature set
    for a given component's model.
    This function is designed to be used consistently in both training and inference.
    """
    processed_df = data_df.copy()

    # Ensure Timestamp is a datetime object for time-series operations
    if 'Timestamp' in processed_df.columns:
        processed_df['Timestamp'] = pd.to_datetime(processed_df['Timestamp'])
    if 'Last_Service_Date' in processed_df.columns:
        processed_df['Last_Service_Date'] = pd.to_datetime(processed_df['Last_Service_Date'])

    # Sort data.
    processed_df = processed_df.sort_values(by=['Vehicle_ID', 'Timestamp']).reset_index(drop=True)

    # Identify numerical features suitable for rolling/lagged calculations.
    numerical_sensor_features = [f for f in component_specific_features_list if
                                 f in processed_df.columns and processed_df[f].dtype in ['int64',
                                                                                         'float64'] and f not in [
                                     'Check_Engine_Light_On']]
    numerical_env_usage_features = [f for f in common_env_usage_features if
                                    f in processed_df.columns and processed_df[f].dtype in ['int64', 'float64']]

    features_for_temporal_engineering = list(set(numerical_sensor_features + numerical_env_usage_features))

    # Apply rolling window and lag operations, grouped by each Vehicle_ID.
    for feature in features_for_temporal_engineering:
        # Rolling mean over the last 4 hours (16 intervals of 15 minutes)
        processed_df[f'{feature}_rolling_mean_4hr'] = processed_df.groupby('Vehicle_ID')[feature].transform(
            lambda x: x.rolling(window=16, min_periods=1).mean())
        # Rolling standard deviation over the last 4 hours
        processed_df[f'{feature}_rolling_std_4hr'] = processed_df.groupby('Vehicle_ID')[feature].transform(
            lambda x: x.rolling(window=16, min_periods=1).std())
        # Value from 1 hour ago (4 intervals back)
        processed_df[f'{feature}_lag_4'] = processed_df.groupby('Vehicle_ID')[feature].transform(lambda x: x.shift(4))

    processed_df = processed_df.ffill().bfill()

    final_features_list = list(
        set(component_specific_features_list + common_env_usage_features + vehicle_metadata_features))

    engineered_features = [col for col in processed_df.columns if '_rolling_' in col or '_lag_' in col]
    final_features_list.extend(engineered_features)

    # Remove any non-feature columns
    non_feature_cols = ['Timestamp', 'Vehicle_ID', 'Last_Service_Date', 'Maintenance_Type'] + component_targets_list
    final_features_list = [f for f in final_features_list if f not in non_feature_cols]

    final_features_list.sort()

    # Return the processed DataFrame and the final list of features
    return processed_df, final_features_list
