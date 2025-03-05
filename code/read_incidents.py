import pandas as pd
import requests
import yaml
import json
from nvd_api import NvdApiClient
import re
import datetime
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nvd_api_key = "3ce37055-92b4-455b-94f5-a52df634f439"
client = NvdApiClient()
client = NvdApiClient(wait_time=1 * 1000, api_key=nvd_api_key)
CVSS_V31_STR = "cvssMetricV31"
CVSS_V30_STR = "cvssMetricV30"

# Define a function to get CVE details from the NVD API
def get_cve_details(cve_id):
    base_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    headers = {
        'apiKey': nvd_api_key
    }
    try:
        request_url = f"{base_url}?cveId={cve_id}"
        print(f"Requesting: {request_url}")
        response = requests.get(request_url, headers=headers, timeout=5)
        if response.status_code == 200:
            # Decode the byte string to a JSON string
            json_str = response.content.decode('utf-8')
            # Parse the JSON string into a Python dictionary
            cve_data = json.loads(json_str)
            # Extract the CVE object
            cve = cve_data.get("vulnerabilities", [{}])[0].get("cve", {})
            # Check for CVSS v3.1 or CVSS v3.0 metrics first
            metric = None
            if CVSS_V31_STR in cve.get("metrics", {}):
                metric = CVSS_V31_STR
            elif CVSS_V30_STR in cve.get("metrics", {}):
                metric = CVSS_V30_STR

            if metric:
                # Extracting relevant CVE details
                cve_description = cve["descriptions"][0].get('value')
                base_score = cve["metrics"][metric][0]["cvssData"].get('baseScore')
                base_severity = cve["metrics"][metric][0]["cvssData"].get('baseSeverity')
                attack_vector = cve["metrics"][metric][0]["cvssData"].get('attackVector')
                attack_complexity = cve["metrics"][metric][0]["cvssData"].get('attackComplexity')
                privileges_required = cve["metrics"][metric][0]["cvssData"].get('privilegesRequired')
                user_interaction = cve["metrics"][metric][0]["cvssData"].get('userInteraction')
                # scope = cve["metrics"][metric][0].get('scope')
                exploitability_score = cve["metrics"][metric][0].get('exploitabilityScore')
                impact_score = cve["metrics"][metric][0].get('impactScore')

                # Extracting weakness_type with a default value
                weakness_type = (
                    cve.get("weaknesses", [{}])[0].get("type", "Unknown")
                )

                # Extracting weakness_description with a default value
                weakness_description = (
                    cve.get("weaknesses", [{}])[0].get("description", [{}])[0].get("value", "No description available")
                )

                # Extracting configuration_vulnerable with a default value
                configuration_vulnerable = (
                    cve.get("configurations", [{}])[0]
                    .get("nodes", [{}])[0]
                    .get("cpeMatch", [{}])[0]
                    .get("vulnerable", False)  # Assume False as a sensible default for vulnerability
                )

                # Extracting and mapping confidentiality, integrity, and availability to 1 or 0
                impact_mapping = {"HIGH": 1, "LOW": 1, "NONE": 0}

                confidentiality = impact_mapping.get(
                    cve["metrics"][metric][0]["cvssData"].get('confidentialityImpact', "NONE"), 0
                )

                integrity = impact_mapping.get(
                    cve["metrics"][metric][0]["cvssData"].get('integrityImpact', "NONE"), 0
                )

                availability = impact_mapping.get(
                    cve["metrics"][metric][0]["cvssData"].get('availabilityImpact', "NONE"), 0
                )

            # Return data as a dictionary
                return {
                    'cve_id': cve_id,
                    'cve_description': cve_description,
                    'base_score': base_score,
                    'base_severity': base_severity,
                    'attack_vector': attack_vector,
                    'attack_complexity': attack_complexity,
                    'privileges_required': privileges_required,
                    'user_interaction': user_interaction,
                    # 'scope': scope,
                    'exploitability_score': exploitability_score,
                    'impact_score': impact_score,
                    'weakness_type': weakness_type,
                    'weakness_description': weakness_description,
                    'configuration_vulnerable': configuration_vulnerable,
                    'confidentiality': confidentiality,
                    'integrity': integrity,
                    'availability': availability
                }
        else:
            print(f"Error fetching {cve_id}: {response.status_code}")
            return get_cve_details(cve_id)
    except Exception as e:
        print(f"An error occurred: {e}")
        return get_cve_details(cve_id)

# Encode all features needed for the training and classification later
def encode_features(data):
    # Map and update each column directly
    # 1. corrupted_asset_criticality
    criticality_mapping = {
        'low': 0,
        'medium': 1,
        'high': 2,
        'critical': 3
    }
    # data['corrupted_asset_criticality'] = data['corrupted_asset_criticality'].map(criticality_mapping)

    # 2. ecu, service, and name
    le_ecu = LabelEncoder()
    data['ecu'] = le_ecu.fit_transform(data['ecu'])
    data['service'] = le_ecu.fit_transform(data['service'])
    data['name'] = le_ecu.fit_transform(data['name'])
    
    # 3. base_severity
    severity_mapping = {'CRITICAL': 3, 'HIGH': 2, 'MEDIUM': 1, 'LOW': 0}
    data['base_severity'] = data['base_severity'].map(severity_mapping)

    # 4. attack_vector
    attack_vector_mapping = {
    'NETWORK': 3,
    'ADJACENT_NETWORK': 2,
    'LOCAL': 1,
    'PHYSICAL': 0
    }
    data['attack_vector'] = data['attack_vector'].map(attack_vector_mapping)

    # 5. attack_complexity
    attack_complexity_mapping = {'LOW': 0, 'HIGH': 1}
    data['attack_complexity'] = data['attack_complexity'].map(attack_complexity_mapping)

    # 6. user_interaction
    user_interaction_mapping = {'NONE': 0, 'REQUIRED': 1}
    data['user_interaction'] = data['user_interaction'].map(user_interaction_mapping)

    # 7. privileges_required
    privileges_required_mapping = {'NONE': 0, 'LOW': 1, 'HIGH': 2}
    data['privileges_required'] = data['privileges_required'].map(privileges_required_mapping)

    # 8. weakness_type
    weakness_type_mapping = {'Primary': 1, 'Secondary': 0}
    data['weakness_type'] = data['weakness_type'].map(weakness_type_mapping)

    # 9. weakness_description
    # Encode CWE for labels
    label_encoder = LabelEncoder()
    data['weakness_description'] = label_encoder.fit_transform(data['weakness_description'])
    
    data = data.drop(['threat_type', 'source',  'description', 'implications'], axis = 1)

    return data

# Read the incidents from a yaml file and create them as a DataFrame
def import_incidents(incident_path="dataset/incidents_2.yaml", vehcile_path="dataset/vehicle_2.yaml"):
    # Read the incidents
    with open(incident_path, 'r') as file:
        incidents_data = yaml.safe_load(file)
    # Read the assets
    with open(vehcile_path, 'r') as file:
        assets_data= yaml.safe_load(file)
    # Convert the incidents data into a DataFrame
    incidents_df = pd.DataFrame(incidents_data['security_incidents'])
    # Convert the assets data into a DataFrame
    assets_df = pd.DataFrame(assets_data['assets'])

    # Select only the specific columns from assets_df that you want to include in the merge
    selected_assets_df = assets_df[['id', 'name', 'service', 'critical', 'ecu', 'firmware']]

    # Rename 'id' in selected_assets_df to 'corrupted_asset_id' for the merge
    selected_assets_df = selected_assets_df.rename(columns={'id': 'corrupted_asset_id'})

    
    # Assign the corrupted_asset_criticality column as a categorical column
    selected_assets_df['critical'] = selected_assets_df['critical'].astype('category')
    # # Initialize columns for each security goal
    # security_goals = ['confidentiality', 'integrity', 'availability']
    # for goal in security_goals:
    #     incidents_df[goal] = incidents_df['affected_security_goals'].apply(lambda goals: 1 if goal in goals else 0)
    
    # # Drop the affected_security_goals column if not needed, in our case we will extract it from the CVE record
    # incidents_df.drop('affected_security_goals', axis=1, inplace=True)

    incidents_df = pd.merge(incidents_df, selected_assets_df, on='corrupted_asset_id', how='left')
    # List to store the CVE data
    cve_data_list = []

    # Iterate through incidents and fetch CVE details
    for index, row in incidents_df.iterrows():
        cve_id = row['cve_id']
        cve_data = get_cve_details(cve_id)
        if cve_data:
            cve_data_list.append(cve_data)
    
    # Convert cve_data_list to a DataFrame
    cve_data_df = pd.DataFrame(cve_data_list)

    incidents_df = pd.merge(incidents_df, cve_data_df, on='cve_id', how='left')
    incidents_df = encode_features(incidents_df)
    incidents_df.to_csv(os.path.join(os.getcwd(), "implementation/Files/mapped_dataset.csv"), index=False)
    
    return incidents_df
