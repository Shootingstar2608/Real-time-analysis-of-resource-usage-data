import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

def plot_attention_weights(attention_weights, seq_length, feature_names, save_path=None):
    """
    Visualize attention weights over the sequence to show which time steps
    contributed most to the anomaly.
    attention_weights: shape (seq_length,)
    """
    plt.figure(figsize=(10, 4))
    sns.heatmap(attention_weights.reshape(1, -1), cmap="YlOrRd", 
                xticklabels=[f"T-{seq_length-i}" for i in range(seq_length)],
                yticklabels=["Attention"], cbar_kws={'label': 'Weight'})
    plt.title("BiLSTM Temporal Attention Weights (Interpretability)")
    plt.xlabel("Time Steps")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def extract_shap_values(model, input_sequence, background_data):
    """
    Extract SHAP values using DeepExplainer (Placeholder/Basic implementation)
    In a real scenario, use:
    import shap
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(input_sequence)
    """
    # For this assignment without shap library installed, we simulate feature importance
    # using gradients (Gradient-based attribution)
    model.eval()
    input_sequence.requires_grad = True
    
    # Forward pass
    reconstruction, attn = model(input_sequence)
    
    # We want to explain the reconstruction error
    error = torch.mean((input_sequence - reconstruction) ** 2)
    error.backward()
    
    # Gradients represent sensitivity of the error to the inputs
    saliency = input_sequence.grad.abs().squeeze(0).cpu().numpy()
    
    # Average over time steps to get feature importance
    feature_importance = np.mean(saliency, axis=0)
    
    # Normalize
    if np.sum(feature_importance) > 0:
        feature_importance = feature_importance / np.sum(feature_importance)
        
    return feature_importance

def plot_feature_importance(feature_importance, feature_names, save_path=None):
    plt.figure(figsize=(8, 5))
    indices = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = feature_importance[indices]
    
    sns.barplot(x=sorted_importance, y=sorted_features, palette="viridis")
    plt.title("Feature Importance (Gradient-based SHAP Proxy)")
    plt.xlabel("Contribution to Anomaly Error")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
