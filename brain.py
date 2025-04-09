# Visualize Functional Connectivity Matrix using Streamlit with Age and Sex Filtering
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import plotting
from nilearn.datasets import fetch_atlas_schaefer_2018
import gdown

# Download data from Google Drive
file_id = "1TcoGm7Ys0X2AHEg1AQR4OjBf1pFZIUaI"
output = "processed_train_data.csv"
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load data
train_data = pd.read_csv("processed_train_data.csv")

# Functional connectivity feature columns
fc_columns = [col for col in train_data.columns if col.startswith('feature_')]
if not fc_columns:
    st.error("No functional connectivity features found. Check dataset column names.")
    st.stop()

train_data = train_data.dropna(subset=fc_columns)

# Compute number of brain regions
num_regions = int((1 + np.sqrt(1 + 8 * len(fc_columns))) // 2)

# Load atlas coordinates
atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
coords = plotting.find_parcellation_cut_coords(labels_img=atlas.maps)
if len(coords) != num_regions:
    st.error(f"Atlas has {len(coords)} regions but FC matrix has {num_regions}. They must match.")
    st.stop()

# Streamlit app layout
st.title("Interactive Brain Connectivity Viewer")

# Age slider
age_range = st.slider("Select Age Range:", min_value=8, max_value=21, value=(10, 14))

# Sex selector
sex_label = st.selectbox("Select Sex:", options=['Male', 'Female'])
sex_bool = True if sex_label == 'Male' else False

# Filter data
subgroup = train_data[
    (train_data['age'] >= age_range[0]) &
    (train_data['age'] <= age_range[1]) &
    (train_data['sex_Male'] == sex_bool)
]

if subgroup.empty:
    st.warning(f"No data for {sex_label} in age range {age_range[0]}–{age_range[1]}.")
    st.stop()

# Compute FC matrix
fc_vector = subgroup[fc_columns].mean().values
fc_matrix = np.zeros((num_regions, num_regions))
fc_matrix[np.triu_indices(num_regions, 1)] = fc_vector
fc_matrix += fc_matrix.T

# View connectome (generate HTML iframe)
view = plotting.view_connectome(
    fc_matrix,
    coords,
    edge_threshold='95%',
    title=f'Connectome - {sex_label}, Age {age_range[0]}–{age_range[1]}',
    node_size=8
)

html_file = "/tmp/streamlit_connectome.html"
view.save_as_html(html_file)

# Display in iframe
st.components.v1.html(view._repr_html_(), height=600, scrolling=True)

# Show top 10 strongest connections in FC matrix
st.subheader("Top 10 strongest functional connections")
upper_tri_indices = np.triu_indices(num_regions, k=1)
fc_values = fc_matrix[upper_tri_indices]
top_indices = np.argsort(np.abs(fc_values))[-10:][::-1]

for i, idx in enumerate(top_indices):
    region1 = upper_tri_indices[0][idx]
    region2 = upper_tri_indices[1][idx]
    strength = fc_matrix[region1, region2]
    st.write(f"{i+1}. Region {region1} - Region {region2}: Strength = {strength:.4f}")
