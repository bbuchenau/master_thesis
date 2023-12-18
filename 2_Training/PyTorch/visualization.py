import matplotlib.pyplot as plt

# Function to visualize evaluation parameters (loss, f1) as line chart.
def visualize_losses(data, label, format):
    plt.figure(figsize=(8, 6))
    plt.plot(data, label=label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.savefig('graphics/' + label + '.svg', format=format)
