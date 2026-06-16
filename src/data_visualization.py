# ==========================================
# File 4: data_visualization.py
# Purpose: Generate and save dataset visualizations using Matplotlib.
# ==========================================

import os
import numpy as np
import matplotlib.pyplot as plt
# Import the load_data function from File 2
from load_dataset import load_data

def generate_visualizations():
    """
    Function to generate key visualizations from the crop recommendation dataset.
    Visualizations are saved locally in the 'dataset/plots' directory.
    Uses only Matplotlib as per requirements.
    """
    # Load dataset
    df = load_data()
    
    # Determine the directory to save the plots
    current_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(current_dir, "..", "dataset", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    print("\n--- Starting Data Visualization ---")
    
    # ==========================================
    # Graph 1: Crop Count Bar Chart
    # Explanation: This bar chart displays the number of samples available for each crop.
    # It helps us verify if the dataset is balanced (equal representation of all crop classes).
    # ==========================================
    crop_counts = df['label'].value_counts()
    
    plt.figure(figsize=(14, 6))
    plt.bar(crop_counts.index, crop_counts.values, color='forestgreen', edgecolor='black', alpha=0.8)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title("Crop Count Distribution (Is the dataset balanced?)", fontsize=14, fontweight='bold')
    plt.xlabel("Crop Name", fontsize=12)
    plt.ylabel("Number of Samples", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    bar_chart_path = os.path.join(plots_dir, "crop_count_distribution.png")
    plt.savefig(bar_chart_path, dpi=100)
    plt.close()
    print(f"Saved Crop Count Bar Chart to: {bar_chart_path}")

    # ==========================================
    # Graph 2: Correlation Heatmap
    # Explanation: Shows linear relationship between all numeric attributes.
    # Value ranges from -1 to +1. High absolute values mean features are strongly correlated.
    # Uses raw Matplotlib logic to build a heatmap.
    # ==========================================
    # Select only numeric features for correlation
    numeric_df = df.drop(columns=['label'])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(9, 7))
    # Render matrix using imshow
    im = ax.imshow(corr_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
    
    # Add Colorbar to show correlation scale
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Correlation Coefficient", rotation=-90, va="bottom")
    
    # Set labels and ticks
    features = list(corr_matrix.columns)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(features, fontsize=10)
    
    # Add numerical labels inside the heatmap grid
    for i in range(len(features)):
        for j in range(len(features)):
            val = corr_matrix.iloc[i, j]
            # Use white text for dark backgrounds (ends of scale) and black for light background
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold')
            
    ax.set_title("Correlation Heatmap of Soil & Weather Features", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    heatmap_path = os.path.join(plots_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=100)
    plt.close()
    print(f"Saved Correlation Heatmap to: {heatmap_path}")

    # ==========================================
    # Graph 3: Histogram for Temperature
    # Explanation: Shows the frequency distribution of the temperature feature.
    # Helps us identify the range of temperatures in which the observations were taken.
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.hist(df['temperature'], bins=20, color='tomato', edgecolor='black', alpha=0.8)
    plt.title("Distribution of Temperature", fontsize=13, fontweight='bold')
    plt.xlabel("Temperature (°C)", fontsize=11)
    plt.ylabel("Frequency / Count", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    temp_hist_path = os.path.join(plots_dir, "temperature_distribution.png")
    plt.savefig(temp_hist_path, dpi=100)
    plt.close()
    print(f"Saved Temperature Histogram to: {temp_hist_path}")

    # ==========================================
    # Graph 4: Histogram for Rainfall
    # Explanation: Shows the frequency distribution of rainfall.
    # Helps identify if the dataset has high-rainfall scenarios (e.g. rice) versus low-rainfall scenarios.
    # ==========================================
    plt.figure(figsize=(8, 5))
    plt.hist(df['rainfall'], bins=20, color='royalblue', edgecolor='black', alpha=0.8)
    plt.title("Distribution of Rainfall", fontsize=13, fontweight='bold')
    plt.xlabel("Rainfall (mm)", fontsize=11)
    plt.ylabel("Frequency / Count", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    rainfall_hist_path = os.path.join(plots_dir, "rainfall_distribution.png")
    plt.savefig(rainfall_hist_path, dpi=100)
    plt.close()
    print(f"Saved Rainfall Histogram to: {rainfall_hist_path}")

    print("\nVisualizations successfully generated and saved inside dataset/plots/")

if __name__ == "__main__":
    generate_visualizations()
