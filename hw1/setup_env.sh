#!/bin/bash
python3 -m venv hw1_venv
source hw1_venv/bin/activate
pip3 install -r requirements.txt
echo "✅ Environment ready! Use ./run.sh to start the program."
