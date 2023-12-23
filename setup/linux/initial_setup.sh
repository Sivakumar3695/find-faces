#!/bin/bash

# Check Python version
required_python_version="3.7"

if [[ "$(python --version | cut -d' ' -f2)" < "$required_python_version" ]]; then
  echo "Error: Python $required_python_version or later is required."
  exit 1
fi

python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

pip install gdown
echo "Downloading model files..."
gdown 1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB -O  ~/.insightface/buffalo_l.zip
unzip ~/.insightface/buffalo_l.zip -d ~/.insightface/models/
rm -rf ~/.insightface/buffalo_l.zip

# Inform the user about the next steps
echo "Setup completed"
