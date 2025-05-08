# plotting_utils.py
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for scripts
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def fig_to_base64(fig):
    """Converts a matplotlib figure to a base64 encoded string."""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig) # Close the figure to free memory
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_regression_true_vs_predicted(y_true, y_pred, title="Regression: True vs. Predicted"):
    """Generates a scatter plot for regression results (true vs. predicted)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, label="Data points")
    min_val = min(y_true.min(), y_pred.min()) if len(y_true) > 0 and len(y_pred) > 0 else 0
    max_val = max(y_true.max(), y_pred.max()) if len(y_true) > 0 and len(y_pred) > 0 else 1
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig_to_base64(fig)

def plot_residuals(y_true, y_pred, title="Regression: Residual Plot"):
    """Generates a residual plot for regression."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax.hlines(0, y_pred.min(), y_pred.max(), colors='r', linestyles='--')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (True - Predicted)")
    ax.set_title(title)
    ax.grid(True)
    return fig_to_base64(fig)


def plot_classification_confusion_matrix(y_true, y_pred_labels, class_names, title="Classification: Confusion Matrix"):
    """Generates a confusion matrix plot for classification results."""
    cm = confusion_matrix(y_true, y_pred_labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    ax.set_title(title)
    return fig_to_base64(fig)

def plot_loss_curve(loss_history, val_loss_history=None, title="Training Loss Curve"):
    """
    Generates a loss curve plot.
    loss_history: List or array of training loss values.
    val_loss_history: Optional list or array of validation loss values.
    """
    if not loss_history: # history might be None or empty
        return None
        
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history, label='Training Loss')
    if val_loss_history:
        ax.plot(val_loss_history, label='Validation Loss')
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig_to_base64(fig)

if __name__ == '__main__':
    print("Plotting utilities module loaded.")
    # (No example execution code here as per user request)