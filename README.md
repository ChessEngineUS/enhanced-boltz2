# Enhanced Boltz-2

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

An enhanced implementation of [Boltz-2](https://github.com/jwohlwend/boltz) with optimized performance, uncertainty quantification, and high-throughput virtual screening capabilities.

## Features

- **Memory-Optimized Inference**: Gradient checkpointing, intelligent caching, and mixed precision (FP16) support
- **Uncertainty Quantification**: Multiple sampling for robust predictions with confidence scores
- **Ensemble Predictions**: Multi-model averaging for improved accuracy
- **High-Throughput Virtual Screening**: Dedicated pipeline for drug discovery workflows
- **Advanced Controllability**: Method conditioning, template-based predictions, and distance constraints
- **Performance Optimizations**: PyTorch 2.0+ compilation, batch optimization, GPU acceleration

## Installation

### Quick Setup (Recommended)

```bash
git clone https://github.com/ChessEngineUS/enhanced-boltz2.git
cd enhanced-boltz2
chmod +x setup.sh
./setup.sh
source boltz2_env/bin/activate
```

### Manual Installation

```bash
python3 -m venv boltz2_env
source boltz2_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install 'boltz[cuda]' -U
```

## Quick Start

```python
from enhanced_boltz2 import *

# Configure
config = EnhancedBoltzConfig(device="cuda", num_samples=5)
predictor = EnhancedBoltz2Predictor(config)

# Define molecules
protein = Molecule(id="A", molecule_type="protein", 
                   sequence="MLARALLLCAVLALSHTANP")
ligand = Molecule(id="LIG", molecule_type="ligand",
                  smiles="CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin

# Predict
request = PredictionRequest(molecules=[protein, ligand], predict_affinity=True)
result = predictor.predict(request)

print(f"Predicted IC50: {result['affinity']['ic50_um']:.3f} µM")
print(f"Confidence: {result['confidence']['overall']:.2f}")
```

## Virtual Screening

```python
from enhanced_boltz2 import VirtualScreeningPipeline

pipeline = VirtualScreeningPipeline(predictor)

hits = pipeline.screen_ligands(
    protein_sequence="MLARALLLCAVLALSHTANP",
    ligand_smiles_list=["CC(=O)OC1=CC=CC=C1C(=O)O", "..."],
    affinity_threshold=-6.0  # IC50 < 1 µM
)

for hit in hits:
    print(f"IC50: {hit['ic50_um']:.3f} µM")
```

## Command Line Usage

```bash
# Single prediction
python enhanced_boltz2.py --mode single --output ./results

# Virtual screening
python enhanced_boltz2.py --mode screen --output ./screening_results
```

## Documentation

See [QUICKSTART.md](QUICKSTART.md) for detailed documentation, examples, and troubleshooting.

## Key Improvements Over Base Boltz-2

| Feature | Base Boltz-2 | Enhanced Boltz-2 |
|---------|-------------|------------------|
| Uncertainty Quantification | ❌ | ✅ |
| Memory Optimization | Basic | Advanced (caching, checkpointing) |
| Batch Processing | Manual | Automated with optimization |
| Virtual Screening Pipeline | ❌ | ✅ |
| PyTorch 2.0 Compilation | ❌ | ✅ |
| Confidence Scoring | Basic | Comprehensive |

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (recommended)
- ~8GB GPU memory (minimum)

## Citation

If you use this implementation, please cite:

```bibtex
@article{passaro2025boltz2,
  title={Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  author={Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  journal={bioRxiv},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original [Boltz-2](https://github.com/jwohlwend/boltz) by MIT
- Built with [PyTorch](https://pytorch.org/)
