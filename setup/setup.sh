# Assuming you have already executed 'conda init;'
#!/bin/bash
set -e

echo "Creating qml-env..."
conda env create -f qiskit-env.yml

echo "Activating qml-env..."
source ~/.bashrc
conda activate qml-env

echo "Installing Qiskit packages..."
pip install qiskit qiskit-machine-learning

echo "Installing QuPCA..."
pip install git+https://github.com/Eagle-quantum/QuPCA.git

echo "Done. Run 'conda activate qml-env' to start."
