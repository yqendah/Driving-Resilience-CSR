import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.impute import SimpleImputer

# Define custom stopword list
custom_stopwords = set(stopwords.words('english')).union({
    "based", "after", "before", "would", "could", "might", "like", "many", "also", "however", "yaw", "without","within","way", "vr","vl","via",
    "tilt", "thus", "thereby", "the",  "soll","soc", "schwimmwinkel", "pre", "performs", "pattern", "path", "party", "open", "of", "non", "in",
    "ihmi", "hr", "hl", "due", "do", "bm"
})

# Define a synonym consolidation dictionary
synonym_mapping = {
    "unresponsive": "unresponsiveness",
    "update": "update",
    "updating": "update",
    "used": "use",
    "using": "use",
    "user": "use",
    "unavailability": "unavailable",
    "unavailable": "unavailable",
    "trigger": "trigger",
    "triggering": "trigger",
    "transmitted": "transmit",
    "transmit": "transmit",
    "transmission": "transmit",
    "trajectory": "trajectory",
    "trajektorie": "trajectory",
    "trajektorienvorverarbeitung": "trajectory",
    "terminating": "terminate",
    "termination": "terminate",
    "targeting": "target",
    "target": "target",
    "tampering": "tamper",
    "tampered": "tamper",
    "tamper": "tamper",
    "status": "state",
    "state": "state",
    "spoofing": "spoof",
    "spoofed": "spoof",
    "signaling": "signal",
    "sensormodul": "sensor",
    "sends": "send",
    "sending": "send",
    "rotational": "rotation",
    "responding": "respond",
    "reporting":"report",
    "reported": "report",
    "replayed":"replay",
    "repeated": "repeat",
    "rendering": "render",
    "rendered": "render",
    "receiver":"receive",
    "receiving":"receive",
    "received": "receive",
    "reading": "read",
    "qualität": "quality",
    "provided": "provide",
    "providing": "provide",
    "properly":"proper",
    "processed":"process",
    "processing": "process",
    "prevents": "prevent",
    "preventing": "prevent",
    "potentially": "potential",
    "positional": "position",
    "positioning": "position",
    "planning": "plan",
    "planned": "plan",
    "physically": "physical",
    "overloading": "overload",
    "overloaded":  "overload",
    "overflowed": "overflow",
    "outputting": "output",
    "operational": "operation",
    "operate": "operation",
    "monitoring": "monitor",
    "modifying": "modify",
    "modifies": "modify",
    "modified": "modify",
    "modification": "modify",
    "misrepresentation": "misrepresent",
    "misleading": "mislead",
    "manipulation": "manipulate",
    "manipulating": "manipulate",
    "manipulates": "manipulate",
    "lighting": "light",
    "leading": "lead",
    "introducing": "introduce",
    "introduces": "introduce",
    "interrupting": "interrupt",
    "interfering": "interfere",
    "interference": "interfere",
    "interfered": "interfere",
    "interception": "intercept",
    "intercepting": "intercept",
    "intercepted": "intercept",
    "injects": "inject",
    "injection": "inject",
    "injecting": "inject",
    "injected": "inject",
    "initiating": "initiate",
    "initialization": "initiate",
    "incorrectly": "incorrect",
    "impacting": "impact",
    "halting": "halt",
    "generating": "generate",
    "generated": "generate",
    "gaining": "gain",
    "gained": "gain",
    "functionality": "function",
    "functional": "function",
    "flooding": "flood",
    "flooded": "flood",
    "feeding": "feed",
    "fed": "feed",
    "falsifying": "false",
    "falsified": "flase",
    "failure": "false",
    "fails": "flase",
    "fail": "false",
    "exposing": "expose",
    "exposure": "expose",
    "exposed": "expose",
    "exploited": "exploit",
    "exploitation": "exploit",
    "exploiting": "exploit",
    "execution": "execute",
    "executed": "execute",
    "estimation": "estimate",
    "encryption": "encrypt",
    "encrypting": "encrypt",
    "ecus": "ecu",
    "dynamikmodul": "dynamic",
    "dynamikmodule": "dynamic",
    "dynamikmodules": "dynamic",
    "disrupted": "disrupt",
    "disrupting": "disrupt",
    "disruption": "disrupt",
    "disrupts": "disrupt",
    "displaying": "display",
    "displayed": "display",
    "disabling": "disable",
    "disables": "disable",
    "delayed": "delay",
    "damaged": "damage",
    "corrupts": "corrupt",
    "corruption": "corrupt",
    "corrupted": "corrupt",
    "correctly": "correct",
    "consumption": "consume",
    "consumer": "consume",
    "computing": "compute",
    "computation": "compute",
    "compromising": "compromise",
    "compromised": "compromise",
    "communication": "communicate",
    "charging": "charge",
    "causing": "cause",
    "caused": "cause",
    "calculating": "calculate",
    "calculation": "calculate",
    "bypassing": "bypass",
    "braking": "brake",
    "blocking": "block",
    "behavior": "behave",
    "attacker": "attack",
    "angular": "angle",
    "alters": "alter",
    "altering": "alter",
    "altered": "alter",
    "alteration": "alter",
    "allows": "allow",
    "allowing": "allow",
    "allowed": "allow",
    "affecting": "affect",
    "activity": "activate",
    "activation": "activate",
    "accuracy": "accurate",
    "accessed": "access",
    "accelerometer": "access",
    "denial of service": "dos",
    "man in the middle": "mitm",
    "man-in-the-middle": "mitm"

}

# Define preprocessing function
def preprocess(text):
    try: 
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Convert to lowercase
        tokens = [token.lower() for token in tokens]
        # Remove stopwords (default + custom)
        tokens = [word for word in tokens if word not in custom_stopwords]
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatize words
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Map synonyms to a single representative term
        tokens = [synonym_mapping.get(word, word) for word in tokens]

        # Join tokens back to a single string
        return ' '.join(tokens)
    except:
        print(f'There is an issue with the input {text}')

def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    data['cluster'] = cluster_labels
    return cluster_labels, kmeans, data

def reduce_to_3d(data):
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)
    return reduced_data

def plot_clusters(cluster_labels, reduced_data, best_clusters):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], 
               c=cluster_labels, cmap='viridis', marker='o', s=50)
    
    ax.set_title(f"Cluster Distribution in 3D Space - {best_clusters}")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    plt.show()

def evaluate_clusters(data, cluster_labels):
    silhouette_avg = silhouette_score(data, cluster_labels)
    return silhouette_avg

def prepare_data(data_cluster):
    data_cluster['text'] = data_cluster['incident_name'] + " " + data_cluster['cve_description']

    data_cluster['clean_text'] = data_cluster['text'].apply(preprocess)

    # Replace NaN with column mean
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    data_cluster = pd.DataFrame(imputer.fit_transform(data_cluster), columns=data_cluster.columns)
    # Vectorize text with TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8) 
    X = vectorizer.fit_transform(data_cluster['clean_text'])

    # Convert TF-IDF matrix to a pandas DataFrame
    features_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    threshold = 0.1 
    non_zero_percentage = (features_df != 0).sum(axis=0) / len(features_df) 
    filtered_features_df = features_df.loc[:, non_zero_percentage > threshold]

    # Add filtered features back to the original DataFrame
    data_cluster = pd.concat([data_cluster, filtered_features_df], axis=1)
    data_cluster = data_cluster.drop(['text', 'clean_text', 'incident_name', 'cve_description'], axis=1)
    return data_cluster

def find_best_clusters(data, min_clusters=2, max_clusters=20):
    data_cluster = data.drop(['incident_id', 'corrupted_asset_id', 'cve_id',  'name', 'weakness_type', 'configuration_vulnerable','attack_vector', 'attack_complexity', 'privileges_required','weakness_description',
                              'user_interaction', 'impact_score','service', 'ecu', 'critical', 'firmware', 'base_score', 'base_severity', 'confidentiality', 'integrity', 'availability'], axis=1)
    data_cluster = prepare_data(data_cluster)
    best_score = -1
    best_clusters = 0
    best_labels = None
    best_kmeans = None
    best_data = None
    print("The columns selected for clustering are:")
    print(data_cluster.columns)
    # Try different cluster counts
    for n_clusters in range(min_clusters, max_clusters + 1):
        cluster_labels, kmeans, data_cluster = perform_clustering(data_cluster, n_clusters)
        silhouette_avg = evaluate_clusters(data_cluster, cluster_labels)
        print(f"Number of Clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
        
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_clusters = n_clusters
            best_labels = cluster_labels
            best_kmeans = kmeans
            best_data = data_cluster

    print(f"Best number of clusters: {best_clusters} with Silhouette Score: {best_score:.3f}")
    print(data_cluster.columns)
    print(data_cluster.shape)
    # Perform PCA and plot the best clustering result
    reduced_data = reduce_to_3d(data_cluster)
    plot_clusters(best_labels, reduced_data, best_clusters)
    
    return best_kmeans, best_labels, best_clusters, best_data
