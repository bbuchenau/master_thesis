import matplotlib.pyplot as plt
import seaborn as sns

# Function to visualize evaluation parameters (loss, f1) as line chart.
def visualize_losses(data, label, format):
    plt.figure(figsize=(8, 6))
    plt.plot(data, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.savefig('graphics/' + label + '.svg', format=format)

# Function to visualize evaluation parameters (TP, FP, TN, FN) as confusion matrix.
def visualize_conf_matrix(data, label, file_format, data_format, type):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt=data_format, cmap="Blues", xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f'Class {label}')
    plt.savefig(f'graphics/conf_matrix_{type}_{label}' + '.svg', format=file_format)
    plt.close()
