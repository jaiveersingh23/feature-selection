def print_results(name, selected_features, accuracy, num_features, time_taken):
    print(f"\n{name}")
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  Features used  : {num_features}")
    print(f"  Time taken     : {time_taken:.2f} seconds")
