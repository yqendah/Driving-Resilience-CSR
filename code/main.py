import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
import read_incidents
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import classify
import cluster
from sklearn.model_selection import train_test_split
import os
import time
import gc
import psutil
from scipy.interpolate import make_interp_spline


# Adjust weights for the equation to calculate the incident ciriticality
def adjust_weights(impact_score):
    if impact_score >= 9.0:
        return 0.6, 0.3, 0.1  # High impact on criticality, lower cluster effect
    elif impact_score >= 7.0:
        return 0.5, 0.3, 0.2  # Medium-high weight
    elif impact_score >= 4.0:
        return 0.4, 0.3, 0.3  # Balanced weight
    else:
        return 0.3, 0.2, 0.5  # Higher weight on asset criticality for low-score cases


# Categorize criticality into discrete values (0, 1, 2, 3)
def categorize_criticality(value):
    if value < 0.1:
        return 0
    elif value < 3.9:
        return 1
    elif value < 6.9:
        return 2
    elif value < 8.9:
        return 3
    else:
        return 4


def load_and_prepare_data(filepath):
    """Loads and preprocesses the incidents dataset."""
    df = pd.read_csv(filepath).dropna()
    return read_incidents.encode_features(df)

def perform_classification(train_test_df):
    """Executes the multi-label classification step."""
    X, y, X_train, X_test, y_train, y_test = classify.prepare_data(train_test_df)
    y1_model, y1_pred = classify.first_layer_classify(X_train, y_train, X_test)
    y2_model, y2_pred = classify.second_layer_classify(X_train, y_train, X_test, y1_pred)
    
    # Evaluate and visualize results
    classify.evaluate_system(y_test, y2_pred)
    classify.plot_metrics(y_test, y2_pred)
    classify.plot_actual_vs_predicted(y_test, y2_pred)
    classify.plot_sum_affected_actual_vs_predicted(y_test, y1_pred)
    
    return (y1_model, y2_model), (y1_pred, y2_pred)

def perform_clustering(train_test_df):
    """Finds the best clusters and processes clustering."""
    best_kmeans, kmeans_labels, kmeans_cluster, clustered_data = cluster.find_best_clusters(train_test_df)
    return best_kmeans, clustered_data

def perform_clustering(row, best_kmeans, buffer, real_time_clusters):
    cluster_start = time.perf_counter()
    row['kmeans_cluster'] = best_kmeans.predict(row)[0]
    real_time_clusters.append(row['kmeans_cluster'])
    cluster_end = time.perf_counter()
    
    buffer.append(row)
    if len(buffer) >= 20:
        X_new = np.vstack(buffer)
        best_kmeans.fit(X_new) 
        buffer.clear()
    
    return row, (cluster_end - cluster_start) * 1000

def perform_classification(row, trained_models, real_time_df):
    accuracy, row, classification_time = classify.real_time_classification(trained_models, row)
    row['sum_affected'] = row[['confidentiality', 'integrity', 'availability']].sum(axis=1)
    
    real_time_df = real_time_df.append(row.drop(['incident_id', 'incident_name', 'cve_id', 'corrupted_asset_id', 'configuration_vulnerable', 'kmeans_cluster',
                   'cve_description', 'name', 'ecu', 'service', 'critical', 'firmware', 'confidentiality', 'weakness_type', 'base_score', 'base_severity', 'integrity', 'availability'], axis=1))
    
    if real_time_df.shape[0] >= 20:
        update_models(real_time_df, trained_models)
        real_time_df = pd.DataFrame()
    
    return row, accuracy, classification_time, real_time_df

def update_models(real_time_df, trained_models):
    X_train_new = real_time_df.drop(['sum_affected'], axis=1)
    y_train_new_first_layer = real_time_df['sum_affected']
    y_train_new_second_layer = real_time_df[['confidentiality', 'integrity', 'availability']]
    
    first_layer_model, second_layer_model = trained_models
    first_layer_model.fit(X_train_new, y_train_new_first_layer)
    second_layer_model.fit(X_train_new, y_train_new_second_layer)

def perform_decision_making(row):
    decision_start = time.perf_counter()
    
    w_security, w_cluster, w_asset = adjust_weights(row['impact_score'].astype(float).values[0])
    criticality = (
        row['critical'] * w_asset +
        ((20 - row['kmeans_cluster']) / 20) * w_cluster +
        (((row['conf_pred'] * 1) + (row['int_pred'] * 3) + (row['av_pred'] * 2)) / 6) * w_security
    ) * 10
    
    w_score = 1 - w_asset
    base_severity_calculated = (((row['base_score'] / 10) * w_score) + (row['critical'] * w_asset)) * 10
    decision_end = time.perf_counter()
    
    return criticality, base_severity_calculated, (decision_end - decision_start) * 1000

def prepare_real_time_eval_df(train_test_df):
    """Prepares the real-time evaluation dataframe by dropping unnecessary columns."""
    drop_columns = [
        'incident_name', 'incident_id', 'cve_id', 'corrupted_asset_id', 'configuration_vulnerable',
        'cve_description', 'name', 'ecu', 'service', 'critical', 'firmware', 'confidentiality',
        'weakness_type', 'base_score', 'base_severity', 'integrity', 'availability'
    ]
    return train_test_df.drop(columns=drop_columns, errors='ignore')


def create_real_time_dataframe(real_time_eval_df, real_time_clusters, criticalities, base_severity_scores, criticality_categories, base_severity_categories, accuracies):
    """ Creates and stores the real-time evaluation dataframe. """
    real_time_dataframe = pd.DataFrame()
    real_time_dataframe["asset_criticality"] = real_time_eval_df["critical"]
    real_time_dataframe["confidentiality"] = real_time_eval_df["confidentiality"]
    real_time_dataframe["integrity"] = real_time_eval_df["integrity"]
    real_time_dataframe["availability"] = real_time_eval_df["availability"]
    real_time_dataframe["cluster"] = real_time_clusters
    real_time_dataframe["criticality"] = criticalities
    real_time_dataframe["base_severity_calculated"] = base_severity_scores
    real_time_dataframe["criticality_category"] = criticality_categories
    real_time_dataframe["base_severity_category"] = base_severity_categories
    real_time_dataframe["accuracy"] = accuracies
    real_time_dataframe["base_severity"] = real_time_eval_df["base_severity"]
    real_time_dataframe["impact_score"] = real_time_eval_df["impact_score"]
    real_time_dataframe["base_score"] = real_time_eval_df["base_score"]
    
    real_time_dataframe.to_csv(os.path.join(os.getcwd(), "implementation/Files/real_time_dataframe_results.csv"), index=False)
    gc.enable()  # Re-enable garbage collection

    return real_time_dataframe

def plot_boxplot(elapsed_times, classification_times, clustering_times, decision_times):
    """ Plots a box plot of the elapsed times for different stages. """
    data = [
        classification_times,
        clustering_times,
        decision_times,
        elapsed_times
    ]
    labels = ["Classification Time", "Clustering Time", "Decision Time", "Total Elapsed Time"]

    plt.figure(figsize=(50, 30))

    # Generate the box plot
    plt.boxplot(data, labels=labels, medianprops={'color': 'black', 'linewidth': 2}, patch_artist=True, boxprops=dict(facecolor="lightgray"))
    plt.xticks(fontsize=35, fontweight='bold')  
    plt.title("Distribution of Elapsed Times for all Incidents", fontsize=100, fontweight='bold')
    plt.ylabel("Elapsed Time (ms)", fontsize=50, fontweight='bold')
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set tick parameters to make the ticks bolder and fonts bigger
    plt.tick_params(axis='x', labelsize=50, width=2)  # X-axis tick settings
    plt.tick_params(axis='y', labelsize=50, width=2)  # Y-axis tick settings

    # Save and show the plot
    plt.savefig('implementation/Files/evaluation_elapsed_time_boxplot.png')
    plt.show()


########################################### Offline Phase ###########################################

########Data Loading and Preparing#############
data_path = os.path.join(os.getcwd(), "Files/mapped_dataset.csv")
incidents_df = load_and_prepare_data(data_path)

# Split the dataset into training/testing (80%) and real-time evaluation (20%)
train_test_df, real_time_eval_df = train_test_split(incidents_df, test_size=0.2, random_state=42)

######## Multi-label Classification Step#############
# Perform classification
trained_models, predictions = perform_classification(train_test_df)

######## Clustering Step#############
# Perform clustering
best_kmeans, clustered_data = perform_clustering(train_test_df)

# Prepare real-time evaluation dataset
real_time_eval_df_to_cluster = prepare_real_time_eval_df(train_test_df)



########################################### Online Phase ###########################################

######## Real-Time Clustering and Evaluation #############
gc.disable()  # Disable garbage collection to reduce variance

# Try to pin execution to a single core (Linux only)
try:
    psutil.Process().cpu_affinity([0])
except AttributeError:
    pass  # Ignore on non-Linux systems

os.environ["OMP_NUM_THREADS"] = "1"
real_time_df = pd.DataFrame()
buffer = []
real_time_clusters = []
elapsed_times, classification_times, decision_times = [], [], []
base_severity_scores, base_severity_categories = [], []
criticalities, criticality_categories, accuracies = [], [], []

for i in range(len(real_time_eval_df)):
    row = real_time_eval_df.iloc[i].to_frame().T
    row_to_cluster = row.drop(['incident_name', 'incident_id', 'cve_id', 'corrupted_asset_id', 'configuration_vulnerable',
                               'cve_description', 'name', 'ecu', 'service', 'critical', 'firmware', 'confidentiality', 'weakness_type', 
                               'base_score', 'base_severity', 'integrity', 'availability'], axis=1)
    
    row_to_cluster = row_to_cluster.reindex(columns=best_kmeans.feature_names_in_, fill_value=0)
    
    # Perform clustering
    row, cluster_time = perform_clustering(row, best_kmeans, buffer, real_time_clusters)
    
    # Perform classification
    row, accuracy, classification_time, real_time_df = perform_classification(row, trained_models, real_time_df)
    
    # Perform decision-making
    criticality, base_severity_calculated, decision_time = perform_decision_making(row)
    
    # Store results
    classification_times.append(classification_time)
    decision_times.append(decision_time)
    elapsed_times.append(cluster_time + classification_time + decision_time)
        
# plot results
plot_boxplot(elapsed_times, classification_times, decision_times)