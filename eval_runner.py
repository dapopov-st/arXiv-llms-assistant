import glob
import subprocess

# Get a list of all config files
config_files = glob.glob('configs/*.json')

# Loop over all config files
for config_file in config_files:
    # Run eval.py with the current config file
    subprocess.run(['python', 'eval.py', '--config', config_file])