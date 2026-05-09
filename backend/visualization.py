import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def save_visualization(original, mask, spread, output_dir, mask_3d=None):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Segmentation Overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(original, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'segmentation.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # 2. Future Spread Map
    plt.figure(figsize=(10, 10))
    plt.imshow(original, cmap='gray')
    plt.imshow(spread, cmap='hot', alpha=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'future_spread.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # 3. GradCAM Explainability (Pseudo-overlay for prototype)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(original, cmap='gray')
    plt.imshow(heatmap, alpha=0.4)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'explainability.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

    # 4. Difference Map
    diff = np.clip(spread.astype(float) - mask.astype(float), 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(original, cmap='gray')
    plt.imshow(diff, cmap='autumn', alpha=0.6)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'difference_map.png'), bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def generate_research_charts(metrics, output_dir, mask_3d=None, brain_vol=None):
    # Loss convergence dummy chart
    fig = go.Figure()
    epochs = list(range(1, 21))
    loss = [0.8 / (i**0.5) for i in epochs]
    fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Loss', line=dict(color='#00f2ff')))
    fig.update_layout(title="Model Convergence", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.write_json(os.path.join(output_dir, 'loss_chart.json'))

    # Metrics Bar Chart
    df = pd.DataFrame({
        'Metric': ['Dice', 'IoU', 'Precision', 'Recall', 'AUC'],
        'Score': [metrics['dice_score'], metrics['iou'], metrics['precision'], metrics['recall'], metrics['auc_score']]
    })
    fig = px.bar(df, x='Metric', y='Score', color='Metric', color_discrete_sequence=px.colors.sequential.Teal)
    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.write_json(os.path.join(output_dir, 'metrics_chart.json'))

    # Growth Chart
    growth_data = [metrics['voxel_volume'], metrics['voxel_volume']*1.1, metrics['voxel_volume']*1.15, metrics['future_volume']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=['Day 0', 'Day 30', 'Day 60', 'Day 90 (Predicted)'], y=growth_data, fill='tozeroy', line=dict(color='#00f2ff')))
    fig.update_layout(title="Projected Tumor Growth", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.write_json(os.path.join(output_dir, 'growth_chart.json'))

    # 3D Tumor Projection
    fig = go.Figure()

    # 1. Add Brain Context (Faint cloud)
    if brain_vol is not None:
        # Sample brain structure (low intensity points)
        bz, by, bx = np.where(brain_vol > 0.1) 
        if len(bx) > 2000:
            idx = np.random.choice(len(bx), 2000, replace=False)
            bx, by, bz = bx[idx], by[idx], bz[idx]
        
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=bz,
            mode='markers',
            name='Brain Structure',
            marker=dict(size=1, color='#444', opacity=0.1)
        ))

    # 2. Add Tumor
    if mask_3d is not None and np.sum(mask_3d) > 0:
        z, y, x = np.where(mask_3d > 0)
        if len(x) > 2000:
            idx = np.random.choice(len(x), 2000, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
            
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name='Tumor Mass',
            marker=dict(size=3, color='#ff0000', opacity=0.8, colorscale='Reds')
        ))
    else:
        # No tumor detected annotation
        fig.add_annotation(text="No significant tumor volume detected in 3D scan",
                          xref="paper", yref="paper", showarrow=False, font=dict(size=16, color="white"))

    fig.update_layout(
        title="3D Tumor Volumetric Projection",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        )
    )
    fig.write_json(os.path.join(output_dir, 'projection_3d.json'))

