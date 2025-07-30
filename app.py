import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import traceback

from utility.feature_engineering import prepare_features_for_component, \
    common_env_usage_features, engine_specific_features, brake_specific_features, \
    tire_specific_features, vehicle_metadata_features


# --- Helper Functions for Styling ---
def get_health_color_hex(score):
    if score >= 0.8: return '#34D399'  # green-500
    if score >= 0.5: return '#FBBF24'  # yellow-500
    return '#EF4444'  # red-500


def get_prob_color_hex(prob):
    if prob <= 0.2: return '#34D399'
    if prob <= 0.5: return '#FBBF24'
    return '#EF4444'


def get_rul_color_hex(rul, threshold_days):
    if rul > threshold_days * 0.5: return '#34D399'  # green-500
    if rul > threshold_days * 0.2: return '#FBBF24'  # yellow-500
    return '#EF4444'  # red-500


# --- Helper Function for Advice Generation ---
def get_advice_streamlit(data, thresholds):
    advice = []
    health_score = data.get('healthScore', 0)
    failure_prob = data.get('failureProb', 0)
    rul = data.get('rul', 0)

    if health_score < thresholds['health_critical']:
        advice.append("Critical Health: Immediate inspection recommended.")
    elif health_score < thresholds['health_warning']:
        advice.append("Degraded Health: Schedule inspection soon.")
    else:
        advice.append("Health: Good. Continue regular monitoring.")

    if failure_prob > thresholds['prob_critical']:
        advice.append("High Failure Risk: Urgent attention required.")
    elif failure_prob > thresholds['prob_warning']:
        advice.append("Moderate Failure Risk: Plan proactive maintenance.")

    if rul < thresholds['rul_critical_days']:
        advice.append("Low RUL: Maintenance is overdue or critically close.")
    elif rul < thresholds['rul_warning_days']:
        advice.append("RUL Approaching: Schedule maintenance in the near future.")

    return advice if advice else ["No immediate action required. Continue monitoring."]


# --- Component-Specific Thresholds for Advice Generation ---
component_thresholds = {
    'engine': {
        'health_warning': 0.70, 'health_critical': 0.50,
        'prob_warning': 0.25, 'prob_critical': 0.50,
        'rul_warning_days': 100, 'rul_critical_days': 40,
        'rul_alert_days': 365
    },
    'brake': {
        'health_warning': 0.60, 'health_critical': 0.40,
        'prob_warning': 0.30, 'prob_critical': 0.60,
        'rul_warning_days': 60, 'rul_critical_days': 15,
        'rul_alert_days': 180
    },
    'tire': {
        'health_warning': 0.50, 'health_critical': 0.20,
        'prob_warning': 0.40, 'prob_critical': 0.70,
        'rul_warning_days': 30, 'rul_critical_days': 7,
        'rul_alert_days': 90
    },
}

# Component target
engine_targets = ['Engine_RUL', 'Engine_TTF', 'Engine_Failure_Probability', 'Engine_Health_Score']
brake_targets = ['Brake_RUL', 'Brake_TTF', 'Brake_Failure_Probability', 'Brake_Health_Score']
tire_targets = ['Tire_RUL', 'Tire_TTF', 'Tire_Failure_Probability', 'Tire_Health_Score']


# --- Load Trained ML Assets (Models, Preprocessors, Feature Lists) ---
@st.cache_resource
def load_all_ml_assets():
    model_dir = 'trained_models'
    loaded_models = {}
    loaded_preprocessors = {}
    loaded_features_lists = {}

    component_names_for_loading = ['engine', 'brake', 'tire']

    for component_lower in component_names_for_loading:
        component_capitalized = component_lower.capitalize()
        try:
            models_path = os.path.join(model_dir, f'{component_lower}_models.pkl')
            preprocessor_path = os.path.join(model_dir, f'{component_lower}_preprocessor.pkl')
            features_list_path = os.path.join(model_dir, f'{component_lower}_features_list.pkl')

            if not os.path.exists(models_path) or \
                    not os.path.exists(preprocessor_path) or \
                    not os.path.exists(features_list_path):
                st.error(f"Error: Missing model files for {component_capitalized} in '{model_dir}'. "
                         "Please ensure you've run the `train_models.py` script first and saved models "
                         f"({models_path}, {preprocessor_path}, {features_list_path}).")
                st.stop()

            # Load the dictionary containing the single multi-output regressor
            loaded_models[component_capitalized] = joblib.load(models_path)
            loaded_preprocessors[component_capitalized] = joblib.load(preprocessor_path)
            loaded_features_lists[component_capitalized] = joblib.load(features_list_path)
        except Exception as e:
            st.error(f"Failed to load ML assets for {component_capitalized}: {e}")
            st.stop()
    return loaded_models, loaded_preprocessors, loaded_features_lists


models, preprocessors, features_lists = load_all_ml_assets()


# --- Prediction Logic ---
def get_predictions_for_uploaded_data(uploaded_df_for_prediction):
    if uploaded_df_for_prediction.empty:
        return None

    if 'Vehicle_ID' not in uploaded_df_for_prediction.columns or 'Timestamp' not in uploaded_df_for_prediction.columns:
        st.error("Internal Error: DataFrame passed to prediction function must contain 'Vehicle_ID' and 'Timestamp'.")
        return None

    uploaded_df_for_prediction['Timestamp'] = pd.to_datetime(uploaded_df_for_prediction['Timestamp'])
    uploaded_df_for_prediction = uploaded_df_for_prediction.sort_values(by=['Vehicle_ID', 'Timestamp']).reset_index(
        drop=True)

    all_predictions = {}
    component_names = ['Engine', 'Brake', 'Tire']

    for component in component_names:
        component_specific_feats = globals()[f'{component.lower()}_specific_features']
        component_targets_list = globals()[f'{component.lower()}_targets']

        # Call prepare_features_for_component with correct arguments
        processed_df_for_component, _ = prepare_features_for_component(
            uploaded_df_for_prediction.copy(),
            component_specific_feats,
            component_targets_list
        )

        latest_processed_data = processed_df_for_component.groupby('Vehicle_ID').last().reset_index()

        expected_features_for_component = features_lists[component]

        for feature in expected_features_for_component:
            if feature not in latest_processed_data.columns:
                st.warning(f"Feature '{feature}' not found in processed data for {component}. Adding with 0.0.")
                latest_processed_data[feature] = 0.0

        X_inference_final = latest_processed_data[expected_features_for_component]

        X_inference_transformed = preprocessors[component].transform(X_inference_final)

        # Access the single multi-output regressor from the loaded dictionary
        component_model = models[component]['multi_output_regressor']
        raw_predictions = component_model.predict(X_inference_transformed)

        for i, vehicle_id in enumerate(latest_processed_data['Vehicle_ID']):
            if vehicle_id not in all_predictions:
                all_predictions[vehicle_id] = {}

            # Extract individual target predictions from the multi-output model's prediction
            # (RUL, TTF, Failure_Probability, Health_Score)
            rul_pred = raw_predictions[i, 0]
            ttf_pred = raw_predictions[i, 1]
            failure_prob_pred = raw_predictions[i, 2]
            health_score_pred = raw_predictions[i, 3]

            all_predictions[vehicle_id][component.lower()] = {
                'rul': max(0, float(rul_pred)),
                'ttf': max(0, float(ttf_pred)),
                'failureProb': np.clip(float(failure_prob_pred), 0.0, 1.0),
                'healthScore': np.clip(float(health_score_pred), 0.0, 1.0)
            }
    return all_predictions


# --- Streamlit UI Setup ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="ICE Vehicle Health Dashboard")

# Custom CSS for dark mode and component card styling
st.markdown("""
    <style>
    .stApp {
        background-color: #1a202c; /* Tailwind's gray-900 */
        color: #e2e8f0; /* Tailwind's gray-200 */
        font-family: sans-serif;
    }
    .stButton > button {
        background-color: #2563eb; /* blue-600 */
        color: white;
        font-weight: bold;
        border-radius: 0.5rem; /* rounded-lg */
        padding: 0.75rem 1.5rem; /* px-6 py-3 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: background-color 0.2s;
        margin-bottom: 2rem; /* Add margin below the button */
    }
    .stButton > button:hover {
        background-color: #1d4ed8; /* blue-700 */
    }
    h1 {
        color: #f7fafc; /* gray-50 */
        text-align: center;
        font-size: 2.25rem; /* text-4xl */
        font-weight: 800; /* font-extrabold */
        margin-bottom: 2.5rem; /* mb-10 */
        letter-spacing: -0.025em; /* tracking-tight */
    }
    h2 {
        color: #90cdf4; /* blue-300 */
        font-size: 1.5rem; /* text-2xl */
        font-weight: 700; /* font-bold */
        margin-bottom: 1rem;
    }
    .component-card {
        background-color: #2d3748; /* gray-700 */
        padding: 1.5rem; /* p-6 */
        border-radius: 0.75rem; /* rounded-xl */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06); /* shadow-lg */
        border: 1px solid #4a5568; /* border-gray-600 */
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        transition: transform 0.2s ease-in-out;
        height: 100%; /* Ensure cards in a row have same height */
    }
    .component-card:hover {
        transform: scale(1.02); /* Slightly larger on hover */
    }
    .icon-circle {
        width: 4rem; /* w-16 */
        height: 4rem; /* h-16 */
        border-radius: 9999px; /* rounded-full */
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        font-size: 2.5rem; /* text-3xl */
    }
    .card-title {
        font-size: 1.25rem; /* text-xl */
        font-weight: 600; /* font-semibold */
        color: #f7fafc; /* gray-100 */
        margin-bottom: 0.75rem;
    }
    .metric-section {
        width: 100%;
        text-align: left;
        margin-bottom: 1rem;
    }
    .metric-label {
        font-size: 0.875rem; /* text-sm */
        font-weight: 500; /* font-medium */
        color: #a0aec0; /* gray-300 */
    }
    .metric-value {
        font-size: 0.875rem; /* text-sm */
        color: #e2e8f0; /* gray-200 */
        margin-top: 0.25rem;
    }
    .progress-bar-bg {
        width: 100%;
        background-color: #4a5568; /* gray-600 */
        border-radius: 9999px; /* rounded-full */
        height: 0.625rem; /* h-2.5 */
    }
    .progress-bar-fill {
        height: 0.625rem;
        border-radius: 9999px;
    }
    .advice-box {
        width: 100%;
        background-color: #1a202c; /* gray-800 */
        padding: 0.75rem; /* p-3 */
        border-radius: 0.5rem; /* rounded-lg */
        border: 1px solid #4a5568; /* border-gray-600 */
        text-align: left;
        margin-top: 1rem;
    }
    .advice-box h4 {
        font-size: 1rem; /* text-md */
        font-weight: 600; /* font-semibold */
        color: #f7fafc; /* gray-100 */
        margin-bottom: 0.5rem;
    }
    .advice-box ul {
        list-style-type: disc;
        margin-left: 1.25rem;
        color: #cbd5e0; /* gray-300 */
        font-size: 0.875rem;
    }
    .advice-box li {
        margin-bottom: 0.25rem;
    }
    .insights-panel {
        margin-top: 2.5rem; /* mt-10 */
        padding: 1.5rem; /* p-6 */
        background-color: #2d3748; /* gray-700 */
        border-radius: 0.5rem; /* rounded-lg */
        border: 1px solid #4a5568; /* border-gray-600 */
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06); /* shadow-inner */
    }
    .insights-panel ul {
        list-style-type: disc;
        margin-left: 1.25rem;
        color: #e2e8f0; /* gray-200 */
        line-height: 1.5;
    }
    .insights-panel p {
        margin-top: 1rem;
        color: #a0aec0; /* gray-400 */
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

st.title("ICE Vehicle Health Dashboard")

# --- File Uploader ---
uploaded_file = st.file_uploader("📁 Upload your Vehicle Sensor CSV file", type=['csv'])

# --- Process Uploaded File and Make Predictions ---
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        # Basic checks for essential columns
        if 'Timestamp' not in uploaded_df.columns:
            st.error(
                "Error: 'Timestamp' column not found in the uploaded CSV. Please ensure your CSV contains this column.")
            st.stop()
        if 'Vehicle_ID' not in uploaded_df.columns:
            st.error(
                "Error: 'Vehicle_ID' column not found in the uploaded CSV. Please ensure your CSV contains this column.")
            st.stop()

        # Convert Timestamp to datetime early
        uploaded_df['Timestamp'] = pd.to_datetime(uploaded_df['Timestamp'])

        st.success("File uploaded successfully!")
        st.write("First 5 rows of uploaded data:")
        st.dataframe(uploaded_df.head())

        # Get unique vehicle IDs for selection
        unique_vehicle_ids = uploaded_df['Vehicle_ID'].unique()
        selected_vehicle_id = st.selectbox("Select a Vehicle ID to view predictions:", unique_vehicle_ids,
                                             key='vehicle_select')

        if st.button(f"Get Predictions for {selected_vehicle_id}", key='predict_button'):
            with st.spinner(f'Getting predictions for {selected_vehicle_id}...'):
                all_vehicle_predictions = get_predictions_for_uploaded_data(uploaded_df.copy())
                if all_vehicle_predictions and selected_vehicle_id in all_vehicle_predictions:
                    current_predictions = all_vehicle_predictions[selected_vehicle_id]
                    st.session_state.current_predictions = current_predictions
                    st.session_state.selected_vehicle_id_display = selected_vehicle_id

                else:
                    st.error("Could not retrieve predictions for the selected vehicle. "
                             "Ensure the selected Vehicle ID exists in the uploaded data.")
                    if 'current_predictions' in st.session_state:
                        del st.session_state['current_predictions']
                    if 'selected_vehicle_id_display' in st.session_state:
                        del st.session_state['selected_vehicle_id_display']

        if 'current_predictions' in st.session_state and 'selected_vehicle_id_display' in st.session_state:
            st.subheader(f"Latest Predictions for Vehicle: {st.session_state.selected_vehicle_id_display}")
            current_predictions = st.session_state.current_predictions


            # --- Component Card Rendering Function (using st.markdown for custom HTML/CSS) ---
            def render_component_card(col, title, icon, data, thresholds):
                with col:
                    html_card = f"""
                    <div class="component-card">
                        <div class="icon-circle" style="background-color: {get_health_color_hex(data['healthScore'])};">
                            <span>{icon}</span>
                        </div>
                        <h3 class="card-title">{title}</h3>
                        <div class="metric-section">
                            <p class="metric-label">Health Score:</p>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="background-color: {get_health_color_hex(data['healthScore'])}; width: {data['healthScore'] * 100}%;"></div>
                            </div>
                            <p class="metric-value">{f"{(data['healthScore'] * 100):.0f}%"}</p>
                        </div>
                        <div class="metric-section">
                            <p class="metric-label">Failure Risk (Next 30 Days):</p>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="background-color: {get_prob_color_hex(data['failureProb'])}; width: {data['failureProb'] * 100}%;"></div>
                            </div>
                            <p class="metric-value">{f"{(data['failureProb'] * 100):.0f}%"}</p>
                        </div>
                        <div class="metric-section">
                            <p class="metric-label">RUL (Days):</p>
                            <div class="progress-bar-bg">
                                <div class="progress-bar-fill" style="background-color: {get_rul_color_hex(data['rul'], thresholds['rul_alert_days'])}; width: {min(100, (data['rul'] / thresholds['rul_alert_days']) * 100)}%;"></div>
                            </div>
                            <p class="metric-value">{f"{data['rul']:.0f} days"}</p>
                        </div>
                        <div class="metric-section">
                            <p class="metric-label">TTF (Days):</p>
                            <p class="metric-value" style="font-weight: 600;">{f"{data['ttf']:.0f} days"}</p>
                        </div>
                        <div class="advice-box">
                            <h4>Advice:</h4>
                            <ul>
                                {"".join([f"<li>{advice}</li>" for advice in get_advice_streamlit(data, thresholds)])}
                            </ul>
                        </div>
                    </div>
                    """
                    st.markdown(html_card, unsafe_allow_html=True)


            # --- Dashboard Layout ---
            col1, col2 = st.columns(2)
            render_component_card(col1, "Engine System", "⚙️", current_predictions['engine'],
                                  component_thresholds['engine'])
            render_component_card(col2, "Brake System", "🛑", current_predictions['brake'],
                                  component_thresholds['brake'])

            col3, col4 = st.columns(2)
            render_component_card(col3, "Tire System", "🛞", current_predictions['tire'],
                                  component_thresholds['tire'])

    except Exception as e:
        st.error(f"An error occurred while processing the CSV file: {e}")
        st.info("Please ensure your CSV file has the correct columns and format as expected by the models.")
        print("\n--- FULL PYTHON TRACEBACK START ---")
        traceback.print_exc()
        print("--- FULL PYTHON TRACEBACK END ---\n")

        if 'current_predictions' in st.session_state:
            del st.session_state['current_predictions']
        if 'selected_vehicle_id_display' in st.session_state:
            del st.session_state['selected_vehicle_id_display']
else:
    st.info("📌 Please upload a CSV file to get started with vehicle health analysis.")
    if 'current_predictions' in st.session_state:
        del st.session_state['current_predictions']
    if 'selected_vehicle_id_display' in st.session_state:
        del st.session_state['selected_vehicle_id_display']

# --- Dashboard Insights Panel ---
st.markdown("""
    <div class="insights-panel">
        <h2>Dashboard Insights:</h2>
        <ul>
            <li><strong>Health Score:</strong> A continuous value (0-100%) indicating the current condition. Higher is better.</li>
            <li><strong>Failure Risk (Next 30 Days):</strong> The probability (0-100%) of failure within the next 30 days. Higher risk means immediate attention might be needed.</li>
            <li><strong>RUL (Remaining Useful Life):</strong> Estimated days remaining before maintenance is required. Plan maintenance as this number gets lower.</li>
            <li><strong>TTF (Time to Failure):</strong> Estimated days until a critical failure event. Indicates urgency for intervention.</li>
        </ul>

    </div>
""", unsafe_allow_html=True)
