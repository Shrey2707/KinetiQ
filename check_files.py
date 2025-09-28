import os

# Get the current folder the script is running from
current_directory = os.getcwd()
print(f"Script is running from this directory:\n{current_directory}\n")

# List all files and folders in that directory
print("Files found in this directory:")
files_in_directory = os.listdir('.') # '.' means the current directory
for f in files_in_directory:
    print(f"- {f}")

# Check specifically for the model file
print("\n--- Diagnosis ---")
model_file = 'mudra_model_augmented.pkl'
if model_file in files_in_directory:
    print(f"✅ SUCCESS: The model file '{model_file}' was found.")
else:
    print(f"❌ ERROR: The model file '{model_file}' was NOT found in the list above.")