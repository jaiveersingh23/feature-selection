import matplotlib.pyplot as plt
import numpy as np

def plot_comparison(results):
    """
    Plots a comparison of the algorithms based on their accuracy.
    
    Arguments:
    - results: List of tuples (name, accuracy, num_features, time_taken)
    """
    # Extracting the names and values from results
    names = [result[0] for result in results]
    accuracies = [result[1] for result in results]
    num_features = [result[2] for result in results]
    times = [result[3] for result in results]
    
    # Plotting accuracy comparison
    plt.figure(figsize=(10, 6))
    plt.barh(names, accuracies, color='skyblue')
    plt.xlabel('Accuracy')
    plt.title('Algorithm Accuracy Comparison')
    plt.grid(True, axis='x')
    plt.show()

    # Plotting feature selection comparison
    plt.figure(figsize=(10, 6))
    plt.barh(names, num_features, color='lightcoral')
    plt.xlabel('Number of Features Selected')
    plt.title('Feature Selection Comparison')
    plt.grid(True, axis='x')
    plt.show()

    # Plotting time taken comparison
    plt.figure(figsize=(10, 6))
    plt.barh(names, times, color='lightgreen')
    plt.xlabel('Time Taken (seconds)')
    plt.title('Time Taken by Algorithms')
    plt.grid(True, axis='x')
    plt.show()

def plot_accuracy_vs_features(results):
    """
    Plots a graph comparing accuracy vs number of features selected.
    
    Arguments:
    - results: List of tuples (name, accuracy, num_features, time_taken)
    """
    names = [result[0] for result in results]
    accuracies = [result[1] for result in results]
    num_features = [result[2] for result in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(num_features, accuracies, color='orange')
    for i, name in enumerate(names):
        plt.text(num_features[i], accuracies[i], name, fontsize=9, ha='right')
    
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Features Selected')
    plt.grid(True)
    plt.show()

def plot_results(results):
    """
    Calls the functions to plot comparison graphs based on algorithm results.
    
    Arguments:
    - results: List of tuples (name, accuracy, num_features, time_taken)
    """
    plot_comparison(results)
    plot_accuracy_vs_features(results)
