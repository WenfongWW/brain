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

# Load data with caching
@st.cache_data
def load_data():
    df = pd.read_csv("processed_train_data.csv")
    fc_cols = [col for col in df.columns if col.startswith('feature_')]
    df = df.dropna(subset=fc_cols)
    return df, fc_cols

train_data, fc_columns = load_data()

# Compute number of brain regions
num_regions = int((1 + np.sqrt(1 + 8 * len(fc_columns))) // 2)

# Load atlas coordinates
@st.cache_resource
def load_coords():
    atlas = fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
    return plotting.find_parcellation_cut_coords(labels_img=atlas.maps)

coords = load_coords()
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
@st.cache_data
def filter_data(df, age_range, sex_bool):
    return df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) & (df['sex_Male'] == sex_bool)]

subgroup = filter_data(train_data, age_range, sex_bool)

if subgroup.empty:
    st.warning(f"No data for {sex_label} in age range {age_range[0]}–{age_range[1]}.")
    st.stop()

# Compute FC matrix
@st.cache_data
def compute_fc_matrix(data):
    fc_vector = data[fc_columns].mean().values
    matrix = np.zeros((num_regions, num_regions))
    matrix[np.triu_indices(num_regions, 1)] = fc_vector
    matrix += matrix.T
    return matrix, fc_vector

fc_matrix, fc_vector = compute_fc_matrix(subgroup)

# Display connectome only on button click
if st.button("Generate Connectome Visualization"):
    view = plotting.view_connectome(
        fc_matrix,
        coords,
        edge_threshold='95%',
        title=f'Connectome - {sex_label}, Age {age_range[0]}–{age_range[1]}',
        node_size=8
    )
    st.components.v1.html(view._repr_html_(), height=1200, scrolling=True)

# Show top 10 changing functional connections
st.subheader("Top 10 Changing Functional Connections by Age Correlation")

# Compute correlation with age for each FC feature
age_filtered = subgroup['age'].values
correlations = [np.corrcoef(age_filtered, subgroup[fc])[0, 1] for fc in fc_columns]
correlations = np.nan_to_num(correlations)

# Find top 10 FC features with highest absolute correlation
top_corr_indices = np.argsort(np.abs(correlations))[-10:][::-1]

row_idx, col_idx = np.triu_indices(num_regions, k=1)
for i, idx in enumerate(top_corr_indices):
    region1 = row_idx[idx]
    region2 = col_idx[idx]
    st.write(f"{i+1}. Region {region1} - Region {region2}: Correlation with age = {correlations[idx]:.4f}")

# Compute and display sex-based difference matrix interactively
st.subheader("Female vs Male Connectivity Differences in Selected Age Range")

female_subgroup = filter_data(train_data, age_range, sex_bool=False)
male_subgroup = filter_data(train_data, age_range, sex_bool=True)

if not female_subgroup.empty and not male_subgroup.empty:
    female_matrix, _ = compute_fc_matrix(female_subgroup)
    male_matrix, _ = compute_fc_matrix(male_subgroup)
    diff_matrix = female_matrix - male_matrix
    view_diff = plotting.view_connectome(
        diff_matrix,
        coords,
        edge_threshold='95%',
        title=f'Difference Connectome (Female - Male), Age {age_range[0]}–{age_range[1]}',
        node_size=8
    )
    st.components.v1.html(view_diff._repr_html_(), height=1200, scrolling=True)
else:
    st.info("Not enough data for both sexes in this age range to show difference connectome.")

# Developmental trends bar chart
st.subheader("Developmental Trends in Connectivity")

def compute_connectivity_change(data, fc_columns, sex_bool):
    early = data[(data['age'] >= 10) & (data['age'] <= 14) & (data['sex_Male'] == sex_bool)][fc_columns].mean()
    mid = data[(data['age'] > 14) & (data['age'] <= 17) & (data['sex_Male'] == sex_bool)][fc_columns].mean()
    late = data[(data['age'] > 17) & (data['age'] <= 21) & (data['sex_Male'] == sex_bool)][fc_columns].mean()
    early_to_mid = (mid - early).mean()
    mid_to_late = (late - mid).mean()
    return early_to_mid, mid_to_late

male_early_mid, male_mid_late = compute_connectivity_change(train_data, fc_columns, True)
female_early_mid, female_mid_late = compute_connectivity_change(train_data, fc_columns, False)

trend_df = pd.DataFrame({
    'Sex': ['Male', 'Male', 'Female', 'Female'],
    'Stage': ['Early to Mid', 'Mid to Late', 'Early to Mid', 'Mid to Late'],
    'Avg Connectivity Change': [male_early_mid, male_mid_late, female_early_mid, female_mid_late]
})

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=trend_df, x='Stage', y='Avg Connectivity Change', hue='Sex', ax=ax)
ax.set_title('Developmental Connectivity Trends by Sex')
st.pyplot(fig)
