import scipy.io
import os

# Define the filename of your .mat file
# Make sure the file is in the same directory as the script, or provide the full path
filename = 'Leukemia_1.mat'

# Check if the file exists
if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found.")
else:
    # Load the .mat file data into a Python dictionary
    # The loadmat function reads the file and returns a dictionary
    mat_data = scipy.io.loadmat(filename)

    # Print the keys (variable names) in the dictionary to see what variables are stored
    print(f"Variables found in '{filename}': {mat_data.keys()}")

    # Iterate over the variables and print their contents
    # You might need to adjust the specific key names based on your file's content
    for var_name, value in mat_data.items():
        # Exclude metadata like __header__, __version__, __globals__
        if not var_name.startswith('__'):
            print(f"\n--- Data for variable: '{var_name}' ---")
            # Using repr() might show more data for large arrays than simple print()
            print(repr(value)) 

