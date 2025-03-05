import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import MultiOutputClassifier
import time

# Function to prepare data
def prepare_data(data):
    data = data.dropna()
    X = data.drop(['incident_id', 'incident_name', 'cve_id', 'corrupted_asset_id', 'configuration_vulnerable', 
                   'cve_description', 'name', 'ecu', 'service', 'critical', 'firmware', 'exploitability_score',
                   'attack_vector', 'attack_complexity', 'privileges_required', 'user_interaction', 'confidentiality', 
                   'weakness_type', 'base_score', 'integrity', 'availability'], axis=1)
    y = data[['confidentiality', 'integrity', 'availability']]
    y['sum_affected'] = y.sum(axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

def real_time_classification(trained_models, incident):
    X_real_time = incident.drop(['incident_id', 'incident_name', 'cve_id', 'corrupted_asset_id', 'configuration_vulnerable',
                                 'cve_description', 'name', 'ecu', 'service', 'critical', 'firmware', 'exploitability_score', 
                                 'attack_vector', 'attack_complexity', 'privileges_required', 'user_interaction', 'kmeans_cluster', 
                                 'sum_affected', 'confidentiality', 'weakness_type', 'base_score', 'integrity', 'availability'], axis=1)
    y_real_time = incident[['confidentiality', 'integrity', 'availability']]
    
    first_layer_model, second_layer_model = trained_models
    accuracies = []
    conf_pred_values, int_pred_values, av_pred_values = [], [], []

    sample = X_real_time.iloc[0].values.reshape(1, -1)
    true_label = y_real_time.iloc[0].values.tolist()

    classification_start = time.perf_counter()
    y1_pred = first_layer_model.predict(sample)[0]
    y2_pred = np.array([y1_pred / y1_pred] * 3) if y1_pred in [0, 3] else second_layer_model.predict(sample)[0]
    classification_end = time.perf_counter()

    accuracies.append(int(accuracy_score(true_label, y2_pred)))
    conf_pred_values.append(y2_pred[0])
    int_pred_values.append(y2_pred[1])
    av_pred_values.append(y2_pred[2])

    incident['conf_pred'] = conf_pred_values
    incident['int_pred'] = int_pred_values
    incident['av_pred'] = av_pred_values
    classification_time = classification_end - classification_start
    return accuracies, incident, classification_time

def first_layer_classify(X_train, y_train, X_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train['sum_affected'])
    return clf, clf.predict(X_test)

def second_layer_classify(X_train, y_train, X_test, y1_pred):
    y_train = y_train[['confidentiality', 'integrity', 'availability']]
    base_clf = RandomForestClassifier(random_state=42)
    multi_clf = MultiOutputClassifier(base_clf)
    multi_clf.fit(X_train, y_train)

    y2_pred = [multi_clf.predict([X_test.iloc[i]])[0] if prediction not in [0, 3] else np.array([prediction] * 3) for i, prediction in enumerate(y1_pred)]
    return multi_clf, y2_pred

def evaluate_system(y_test, final_predictions):
    y_test_flat = y_test[['confidentiality', 'integrity', 'availability']].values
    predictions_flat = np.array(final_predictions)

    metrics = {goal: {
        'accuracy': accuracy_score(y_test_flat[:, i], predictions_flat[:, i]),
        'precision': precision_score(y_test_flat[:, i], predictions_flat[:, i], zero_division=0),
        'recall': recall_score(y_test_flat[:, i], predictions_flat[:, i], zero_division=0),
        'f1_score': f1_score(y_test_flat[:, i], predictions_flat[:, i], zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_flat[:, i], predictions_flat[:, i]).tolist()
    } for i, goal in enumerate(['confidentiality', 'integrity', 'availability'])}

    y_test_flat_combined = y_test_flat.flatten()
    predictions_flat_combined = predictions_flat.flatten()

    system_metrics = {
        'accuracy': accuracy_score(y_test_flat_combined, predictions_flat_combined),
        'precision': precision_score(y_test_flat_combined, predictions_flat_combined),
        'recall': recall_score(y_test_flat_combined, predictions_flat_combined),
        'f1_score': f1_score(y_test_flat_combined, predictions_flat_combined),
        'confusion_matrix': confusion_matrix(y_test_flat_combined, predictions_flat_combined).tolist()
    }

    print_metrics(metrics)
    print_system_metrics(system_metrics)

def print_metrics(metrics):
    for goal, goal_metrics in metrics.items():
        print(f"\n{goal.capitalize()} Metrics:")
        for metric, value in goal_metrics.items():
            print(f"  {metric.capitalize()}: {value:.2f}")

def print_system_metrics(system_metrics):
    print("\nOverall System Performance:")
    for metric, value in system_metrics.items():
        print(f"  {metric.capitalize()}: {value:.2f}")

def plot_metrics(y_test, final_predictions, labels=['confidentiality', 'integrity', 'availability']):

    # Convert to numpy arrays for easier handling
    y_test_flat = y_test[['confidentiality', 'integrity', 'availability']].values
    predictions_flat = np.array(final_predictions)
    
    # Create subplots for confusion matrices and ROC curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Initialize lists to store the individual metrics (accuracy, precision, recall, F1)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
    
    # Plot confusion matrix and ROC curve for each label
    for i, goal in enumerate(labels):
        y_true = y_test_flat[:, i]
        y_pred = predictions_flat[:, i]
        
        # Compute and plot confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='gray', xticklabels=['Pred: 0', 'Pred: 1'], 
                    yticklabels=['True: 0', 'True: 1'], ax=axes[0, i])
        axes[0, i].set_title(f'Confusion Matrix: {goal.capitalize()}')
        axes[0, i].set_xlabel('Predicted')
        axes[0, i].set_ylabel('Actual')
        
        plt.savefig('implementation/Files/confusion_matrix_goals.png')
        
        # Calculate and store metrics for the current label
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)
        
        # Plot ROC curve for the current label
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        axes[1, i].plot(fpr, tpr, color='gray', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, i].plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
        axes[1, i].set_title(f'ROC Curve: {goal.capitalize()}')
        axes[1, i].set_xlabel('False Positive Rate')
        axes[1, i].set_ylabel('True Positive Rate')
        axes[1, i].legend(loc='lower right')
        plt.savefig('implementation/Files/roc_curve_goals.png')
    # Plot the overall confusion matrix (flattened)
    y_true_flattened = y_test_flat.flatten()
    y_pred_flattened = predictions_flat.flatten()
    cm_system = confusion_matrix(y_true_flattened, y_pred_flattened)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_system, annot=True, fmt='d', cmap='gray', xticklabels=['Pred: 0', 'Pred: 1'], 
                yticklabels=['True: 0', 'True: 1'])
    plt.title('Overall Confusion Matrix (All Labels)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('implementation/Files/overall_confusion_matrix.png')
    
    # Compute and plot the overall ROC curve
    fpr, tpr, _ = roc_curve(y_true_flattened, y_pred_flattened)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='gray', lw=2, label=f'Overall ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guess
    plt.title('Overall ROC Curve (All Labels)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('implementation/Files/overall_roc_curve.png')
    # plt.savefig('')
    # Show all plots
    plt.tight_layout()
    plt.show()

def plot_actual_vs_predicted(y_test, final_predictions, labels=['confidentiality', 'integrity', 'availability']):

    # Convert to numpy arrays for easier handling
    y_test_flat = y_test[['confidentiality', 'integrity', 'availability']].values
    predictions_flat = np.array(final_predictions)
    
    # Create a subplot for each label
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, goal in enumerate(labels):
        # Get the actual and predicted values for the current label
        y_true = y_test_flat[:, i]
        y_pred = predictions_flat[:, i]
        
        # Plot actual values
        axes[i].plot(y_true, label='Actual', color='black', linestyle='-', marker='o', markersize=4)
        
        # Plot predicted values
        axes[i].plot(y_pred, label='Predicted', color='gray', linestyle='--', marker='x', markersize=4)
        
        # Add labels, title, and legend
        axes[i].set_title(f'Actual vs Predicted for {goal.capitalize()}')
        axes[i].set_xlabel('Sample Index')
        axes[i].set_ylabel('Label Value')
        axes[i].legend(loc='best')

    plt.savefig(f'implementation/Files/Actual_vs_Predicted_for_goals.png')
    plt.tight_layout()
    plt.show()


def plot_sum_affected_actual_vs_predicted(y_test, y_pred, label='sum_affected'):

    # Extract the actual sum_affected values
    y_true = y_test[label].values
    # Convert predicted values to numpy array
    y_pred = np.array(y_pred)
    
    # Plot actual values vs predicted values as line charts
    plt.figure(figsize=(10, 6))
    
    # Plot actual values
    plt.plot(y_true, label='Actual', color='black', linestyle='-', marker='o', markersize=4)
    
    # Plot predicted values
    plt.plot(y_pred, label='Predicted', color='gray', linestyle='--', marker='x', markersize=4)
    
    # Add labels, title, and legend
    plt.title(f'Actual vs Predicted for Sum Affected of Security Goals')
    plt.xlabel('Sample Index')
    plt.ylabel('Sum Affected Value')
    plt.legend(loc='best')
    plt.savefig('implementation/Files/Actual_vs_Predicted_for_Sum_Affected_of_Security_goals.png')
    
    plt.tight_layout()
    plt.show()

def plot_real_time(real_time_accuracies):
    # Plot real-time evaluation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(real_time_accuracies)), real_time_accuracies, marker='.', label='Real Time Accuracy', color='gray')
    plt.xlabel('Time Step')
    plt.ylabel('Accuracy')
    plt.title('Real-time Evaluation: Accuracy for Each Record')
    plt.savefig('implementation/Files/real_time_eval.png')
    plt.legend()
    plt.grid(True)
    plt.show()