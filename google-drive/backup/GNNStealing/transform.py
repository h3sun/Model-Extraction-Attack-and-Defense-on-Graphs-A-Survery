import yaml

# Read the original file
with open('requirments-2.yml', 'r') as f:
    lines = f.readlines()

# Prepare the new YAML structure
env = {
    'name': 'myenv',
    'channels': ['conda-forge'],
    'dependencies': []
}

# Process each line and convert it into the YAML format
for line in lines:
    if line.strip() and not line.startswith('#'):
        parts = line.split()
        name = parts[0]
        version = parts[1]
        build = parts[2]
        # Check if a channel is provided
        if len(parts) > 3:
            channel = parts[3]
            dependency = f'{name}={version}={build}'
            if channel not in env['channels']:
                env['channels'].append(channel)
        else:
            dependency = f'{name}={version}={build}'
        env['dependencies'].append(dependency)

# Write the new YAML content to a file
with open('environment.yml', 'w') as f:
    yaml.dump(env, f)

print("Converted requirements.txt to environment.yml")

# Now you can run the following command to create the environment:
# conda env create -f environment.yml