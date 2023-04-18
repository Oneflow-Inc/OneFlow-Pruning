import os
import subprocess

# Define the path of the script to be tested
script_path = "./"

# Get all file names in the path
file_names = os.listdir(script_path)

# Iterate through all file names
for file_name in file_names:
    # If it is a script file and starts with "test_"
    if file_name.endswith(".py") and file_name.startswith("test_") and file_name != "test_all.py":
        # Concatenate the script file path
        script_file = os.path.join(script_path, file_name)
        # Execute the script file
        try:
            subprocess.run(f"python {script_file}", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"{file_name=} executed successfully")
            with open('log.txt', 'a') as f:
                f.write(f"{file_name} executed successfully\n")
        except:
            print(f"{file_name=} execution failed")
            with open('log.txt', 'a') as f:
                f.write(f"{file_name} executed failed!!\n")

