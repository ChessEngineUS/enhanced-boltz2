#!/bin/bash
# Enhanced Boltz-2 Setup Script

echo "================================================"
echo "  Enhanced Boltz-2 Installation"
echo "================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv boltz2_env

# Activate environment
echo "Activating environment..."
source boltz2_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Boltz-2
echo "Installing Boltz-2..."
pip install 'boltz[cuda]' -U

# Download model weights
echo "Downloading Boltz-2 model weights..."
python3 -c "from boltz.main import setup_model; setup_model('boltz2')"

echo ""
echo "================================================"
echo "  Installation Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  source boltz2_env/bin/activate"
echo ""
echo "To run a prediction:"
echo "  python enhanced_boltz2.py --mode single"
echo ""
