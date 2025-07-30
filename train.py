# Import Libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Load Dataset
df = pd.read_csv('./data/ice_synthetic_nigeria_data_focuse

# Data Cleaning
print(df.duplicated().sum())
df.dropna(axis=0, inplace=True)
df['Timestamp']  = pd.to_datetime(df['Timestamp'])

# Define Problem-Specific Features and Targets

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

engine_targets = [
    'Engine_RUL', 'Engine_TTF', 'Engine_Failure_Probability', 'Engine_Health_Score'
]
brake_targets = [
    'Brake_RUL', 'Brake_TTF', 'Brake_Failure_Probability', 'Brake_Health_Score'
]
tire_targets = [
    'Tire_RUL', 'Tire_TTF', 'Tire_Failure_Probability', 'Tire_Health_Score'
]

vehicle_metadata_features = ['Vehicle_Make', 'Vehicle_Model']

# Data Preprocessing & Feature Engineering Function
def prepare_features_for_component(data_df, component_specific_features_list, component_targets_list):
    processed_df = data_df.copy()

    numerical_sensor_features = [f for f in component_specific_features_list if processed_df[f].dtype in ['int64', 'float64'] and f not in ['Check_Engine_Light_On']]
    numerical_env_usage_features = [f for f in common_env_usage_features if processed_df[f].dtype in ['int64', 'float64']]

    features_for_temporal_engineering = list(set(numerical_sensor_features + numerical_env_usage_features))

    for feature in features_for_temporal_engineering:
        processed_df[f'{feature}_rolling_mean_4hr'] = processed_df.groupby('Vehicle_ID')[feature].transform(lambda x: x.rolling(window=16, min_periods=1).mean())
        processed_df[f'{feature}_rolling_std_4hr'] = processed_df.groupby('Vehicle_ID')[feature].transform(lambda x: x.rolling(window=16, min_periods=1).std())
        processed_df[f'{feature}_lag_4'] = processed_df.groupby('Vehicle_ID')[feature].transform(lambda x: x.shift(4))

    processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')

    final_features_list = list(set(component_specific_features_list + common_env_usage_features + vehicle_metadata_features))

    engineered_features = [col for col in processed_df.columns if '_rolling_' in col or '_lag_' in col]
    final_features_list.extend(engineered_features)

    non_feature_cols = ['Timestamp', 'Vehicle_ID', 'Last_Service_Date', 'Maintenance_Type'] + component_targets_list
    final_features_list = [f for f in final_features_list if f not in non_feature_cols]

    final_features_list.sort()

    processed_df = processed_df[final_features_list + component_targets_list ]

    return processed_df, final_features_list

engine_df, engine_features = prepare_features_for_component(df, engine_specific_features, engine_targets)
brake_df, brake_features = prepare_features_for_component(df, brake_specific_features, brake_targets)
tire_df, tire_features = prepare_features_for_component(df, tire_specific_features, tire_targets)

# Data Splitting
X_engine = engine_df.drop(engine_targets, axis=1)
y_engine = engine_df[engine_targets]

categorical_features = ['Vehicle_Make', 'Vehicle_Model']
categorical_indices_engine = [X_engine.columns.get_loc(col) for col in categorical_features]

e_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_indices_engine)], remainder='passthrough')
X_engine = e_encoder.fit_transform(X_engine)

X_train_engine, X_test_engine, y_train_engine, y_test_engine = train_test_split(X_engine,y_engine, random_state=42, test_size=0.2)

X_brake = brake_df.drop(brake_targets, axis=1)
y_brake = brake_df[brake_targets]

categorical_indices_brake = [X_brake.columns.get_loc(col) for col in categorical_features]

b_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_indices_brake)], remainder='passthrough')
X_brake = b_encoder.fit_transform(X_brake)

X_train_brake, X_test_brake, y_train_brake, y_test_brake = train_test_split(X_brake,y_brake, random_state=42, test_size=0.2)

X_tire = tire_df.drop(tire_targets, axis=1)
y_tire = tire_df[tire_targets]

categorical_indices_tire = [X_tire.columns.get_loc(col) for col in categorical_features]

t_encoder = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_indices_tire)], remainder='passthrough')
X_tire = t_encoder.fit_transform(X_tire)

X_train_tire, X_test_tire, y_train_tire, y_test_tire = train_test_split(X_tire,y_tire, random_state=42, test_size=0.2)

# Model Training

engine_model = RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=-1)
engine_model.fit(X_train_engine, y_train_engine)

brake_model = RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=-1)
brake_model.fit(X_train_brake, y_train_brake)

tire_model = RandomForestRegressor(n_estimators=5, random_state=42, n_jobs=-1)
tire_model.fit(X_train_tire, y_train_tire)

# Model Evaluation
y_pred = engine_model.predict(X_test_engine)

mae = mean_absolute_error(y_test_engine, y_pred)
mse = mean_squared_error(y_test_engine, y_pred)
r2 = r2_score(y_test_engine, y_pred)

print("\nEvaluation metrics for Engine Component")
print(f"MAE for Engine Component: {mae:.2f}")
print(f"MSE for Engine Component: {mse:.2f}")
print(f"R2 Engine Component: {r2:.2f}")

y_pred = brake_model.predict(X_test_brake)

mae = mean_absolute_error(y_test_brake, y_pred)
mse = mean_squared_error(y_test_brake, y_pred)
r2 = r2_score(y_test_brake, y_pred)

print("\nEvaluation metrics for Brake Component")
print(f"MAE for Brake Component: {mae:.2f}")
print(f"MSE for Brake Component: {mse:.2f}")
print(f"R2 Brake Component: {r2:.2f}")

y_pred = tire_model.predict(X_test_tire)

mae = mean_absolute_error(y_test_tire, y_pred)
mse = mean_squared_error(y_test_tire, y_pred)
r2 = r2_score(y_test_tire, y_pred)

print("\nEvaluation metrics for Tire Component")
print(f"MAE for Tire Component: {mae:.2f}")
print(f"MSE for Tire Component: {mse:.2f}")
print(f"R2 Tire Component: {r2:.2f}")

# Create a directory to save models and encoders if it doesn't exist
os.makedirs('trained_models', exist_ok=True)

# Save encoders
joblib.dump(e_encoder, filename='trained_models/engine_preprocessor.pkl')
joblib.dump(b_encoder, filename='trained_models/brake_preprocessor.pkl')
joblib.dump(t_encoder, filename='trained_models/tire_preprocessor.pkl')

# Save trained models
joblib.dump({'multi_output_regressor': engine_model}, filename='trained_models/engine_models.pkl')
joblib.dump({'multi_output_regressor': brake_model}, filename='trained_models/brake_models.pkl')
joblib.dump({'multi_output_regressor': tire_model}, filename='trained_models/tire_models.pkl')

# Save the feature lists
joblib.dump(engine_features, filename='trained_models/engine_features_list.pkl')
joblib.dump(brake_features, filename='trained_models/brake_features_list.pkl')
joblib.dump(tire_features, filename='trained_models/tire_features_list.pkl')

print("\nEncoders, models, and feature lists saved successfully!")
