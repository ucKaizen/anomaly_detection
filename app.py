import gradio as gr
import os
import numpy as np
import cv2
import random
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input  # Add this import
from tensorflow.keras.models import Model
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import matplotlib.pyplot as plt

# Paths (adjust as needed)
dataset_path = "data"
basmati_path = os.path.join(dataset_path, "basmati")
jasmine_path = os.path.join(dataset_path, "jasmine")

# Load and preprocess images
def load_images_from_folder(folder, label, limit=None):
    images = []
    filenames = os.listdir(folder)
    if limit:
        filenames = random.sample(filenames, limit)
    img_data = []
    for filename in filenames:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img.astype(np.float32))  # Now this will work
            images.append(img)
            img_data.append((img, filename, label))
    return np.array(images), img_data

# Load data
all_basmati_images, all_basmati_data = load_images_from_folder(basmati_path, label=0)
jasmine_images, jasmine_data = load_images_from_folder(jasmine_path, label=1, limit=None)

# Training and test sets
basmati_train_count = int(0.2 * len(all_basmati_images))
basmati_train_indices = random.sample(range(len(all_basmati_images)), basmati_train_count)
X_train = np.array([all_basmati_images[i] for i in basmati_train_indices])
train_data = [all_basmati_data[i] for i in basmati_train_indices]

basmati_test_count = 200
basmati_test_indices = random.sample(range(len(all_basmati_images)), basmati_test_count)
X_test_basmati = np.array([all_basmati_images[i] for i in basmati_test_indices])
test_data_basmati = [all_basmati_data[i] for i in basmati_test_indices]

jasmine_test_count = 10
jasmine_test_images, jasmine_test_data = load_images_from_folder(jasmine_path, label=1, limit=jasmine_test_count)

X_test = np.concatenate([X_test_basmati, jasmine_test_images], axis=0)
test_data = test_data_basmati + jasmine_test_data
y_test = np.array([0] * len(X_test_basmati) + [1] * len(jasmine_test_images))

# Feature extraction
base_model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(images, batch_size=16):
    return feature_extractor.predict(images, batch_size=batch_size, verbose=1)

X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)

# PCA
pca = PCA(n_components=50)
X_train_reduced = pca.fit_transform(X_train_features)
X_test_reduced = pca.transform(X_test_features)

# Main anomaly detection function
def run_anomaly_detection(mode, model_name, contamination, n_estimators, n_neighbors, nu):
    # Adjust training data for semi-supervised mode
    if mode == "Semi-supervised":
        # Add a small portion of Jasmine to training (e.g., 5 images)
        jasmine_train_count = 5
        jasmine_train_images, jasmine_train_data = load_images_from_folder(jasmine_path, label=1, limit=jasmine_train_count)
        X_train_semi = np.concatenate([X_train, jasmine_train_images], axis=0)
        X_train_semi_features = extract_features(X_train_semi)
        X_train_semi_reduced = pca.transform(X_train_semi_features)
    else:
        X_train_semi_reduced = X_train_reduced

    # Initialize model based on selection
    if model_name == "IForest":
        outlier_detector = IForest(contamination=contamination, n_estimators=int(n_estimators))
    elif model_name == "LOF":
        outlier_detector = LOF(contamination=contamination, n_neighbors=int(n_neighbors))
    else:  # OCSVM
        outlier_detector = OCSVM(contamination=contamination, nu=nu)

    # Fit and predict
    outlier_detector.fit(X_train_semi_reduced)
    predictions = outlier_detector.predict(X_test_reduced)

    # Evaluation
    report = classification_report(y_test, predictions)
    try:
        auc_score = roc_auc_score(y_test, predictions)
        auc_text = f"AUC Score: {auc_score:.4f}"
    except:
        auc_text = "AUC Score could not be calculated."

    # Outlier filenames
    outlier_indices = np.where(predictions == 1)[0]
    outlier_list = []
    for idx in outlier_indices:
        img, filename, label = test_data[idx]
        rice_type = "Jasmine" if label == 1 else "Basmati"
        outlier_list.append(f"Filename: {filename}, Actual Label: {rice_type}")
    outlier_text = "\n".join(outlier_list) if outlier_list else "No outliers detected."

    # PCA Visualization (2D)
    pca_vis = PCA(n_components=2)
    X_test_2d = pca_vis.fit_transform(X_test_features)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1], c='blue', label='Basmati', alpha=0.6, s=40)
    plt.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1], c='red', label='Jasmine', alpha=0.6, s=40)
    plt.scatter(X_test_2d[outlier_indices, 0], X_test_2d[outlier_indices, 1], 
                facecolors='none', edgecolors='black', linewidths=1.5, label='Outliers', s=80)
    plt.title("PCA Projection with Outliers")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return report, auc_text, outlier_text, plt

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown("## Anomaly Detection Playground")
    
    with gr.Row():
        mode = gr.Dropdown(["Unsupervised", "Semi-supervised"], label="Mode")
        model_name = gr.Dropdown(["IForest", "LOF", "OCSVM"], label="Model")
    
    with gr.Row():
        contamination = gr.Slider(0, 0.25, value=0.05, step=0.01, label="Contamination")
        n_estimators = gr.Slider(100, 299, value=100, step=10, label="N Estimators (IForest)")
        n_neighbors = gr.Slider(5, 50, value=20, step=1, label="N Neighbors (LOF)")
        nu = gr.Slider(0, 1, value=0.1, step=0.01, label="Nu (OCSVM)")

    submit_btn = gr.Button("Run Detection")
    
    with gr.Row():
        report_output = gr.Textbox(label="Classification Report")
        auc_output = gr.Textbox(label="AUC Score")
    
    outlier_output = gr.Textbox(label="Detected Outliers")
    plot_output = gr.Plot(label="PCA Projection")

    submit_btn.click(
        fn=run_anomaly_detection,
        inputs=[mode, model_name, contamination, n_estimators, n_neighbors, nu],
        outputs=[report_output, auc_output, outlier_output, plot_output]
    )

interface.launch()