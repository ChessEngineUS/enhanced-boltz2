# Enhanced Boltz-2 Quick Start Guide

## Installation

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
source boltz2_env/bin/activate
```

### Option 2: Manual Installation
```bash
# Create environment
python3 -m venv boltz2_env
source boltz2_env/bin/activate

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install 'boltz[cuda]' -U
```

## Usage

### Single Prediction
```python
from enhanced_boltz2 import (
    EnhancedBoltzConfig, 
    EnhancedBoltz2Predictor,
    Molecule, 
    PredictionRequest
)

# Configure
config = EnhancedBoltzConfig(
    device="cuda",
    num_samples=5,
    predict_affinity=True
)

# Create predictor
predictor = EnhancedBoltz2Predictor(config)

# Define molecules
protein = Molecule(
    id="A",
    molecule_type="protein",
    sequence="MLARALLLCAVLALSHTANP"
)

ligand = Molecule(
    id="LIG",
    molecule_type="ligand",
    smiles="CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
)

# Create request
request = PredictionRequest(
    molecules=[protein, ligand],
    predict_affinity=True
)

# Run prediction
result = predictor.predict(request)
print(f"Predicted IC50: {result['affinity']['ic50_um']:.3f} µM")
```

### Virtual Screening
```python
from enhanced_boltz2 import VirtualScreeningPipeline

pipeline = VirtualScreeningPipeline(predictor)

# Screen multiple ligands
protein_seq = "MLARALLLCAVLALSHTANP"
ligand_smiles = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
]

hits = pipeline.screen_ligands(
    protein_sequence=protein_seq,
    ligand_smiles_list=ligand_smiles,
    affinity_threshold=-6.0  # IC50 < 1 µM
)

for hit in hits:
    print(f"SMILES: {hit['smiles']}")
    print(f"IC50: {hit['ic50_um']:.3f} µM")
```

### Command Line Usage
```bash
# Single prediction
python enhanced_boltz2.py --mode single --output ./results

# Virtual screening
python enhanced_boltz2.py --mode screen --output ./screening_results

# Custom input
python enhanced_boltz2.py --input my_complex.yaml --output ./output
```

## Key Features

### 1. Memory-Optimized Inference
- Gradient checkpointing for reduced memory
- Automatic batch size optimization
- Efficient caching of intermediate results

### 2. Uncertainty Quantification
- Multiple sampling for robust predictions
- Confidence scores for structure and affinity
- Standard deviation estimates

### 3. Ensemble Predictions
- Multi-model averaging
- Improved accuracy over single models
- Automated consensus building

### 4. High-Throughput Screening
- Batch processing of multiple complexes
- Automatic hit identification
- Ranking by binding affinity

### 5. Advanced Controllability
- Method conditioning (X-ray, NMR, MD)
- Template-based predictions
- Distance constraints

## Performance Tips

### GPU Optimization
```python
config = EnhancedBoltzConfig(
    device="cuda",
    dtype="float16",  # Mixed precision
    compile_model=True,  # PyTorch 2.0 compilation
    gradient_checkpointing=True
)
```

### Memory Management
```python
config = EnhancedBoltzConfig(
    memory_efficient=True,
    max_batch_size=8,
    num_recycles=4  # Reduce for speed
)
```

### Batch Processing
```python
# Process multiple complexes efficiently
requests = [create_request(seq, lig) for seq, lig in zip(proteins, ligands)]
results = predictor.batch_predict(requests)
```

## Output Format

```json
{
  "structure": {
    "coordinates": [[x, y, z], ...],
    "coordinates_std": [[sx, sy, sz], ...]
  },
  "affinity": {
    "value": -7.2,
    "std": 0.15,
    "confidence": 0.89,
    "ic50_um": 0.063
  },
  "confidence": {
    "structure_uncertainty": 0.12,
    "affinity_uncertainty": 0.15,
    "overall": 0.87
  },
  "metadata": {
    "device": "cuda",
    "num_recycles": 4,
    "num_samples": 5
  }
}
```

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size and samples
config.max_batch_size = 4
config.num_samples = 3
config.num_recycles = 3
```

### Slow Predictions
```python
# Enable optimizations
config.compile_model = True  # Requires PyTorch 2.0+
config.dtype = "float16"
config.memory_efficient = True
```

### Import Errors
```bash
# Reinstall Boltz-2
pip uninstall boltz
pip install 'boltz[cuda]' -U
```

## Citation

If you use this enhanced implementation, please cite:

```bibtex
@article{passaro2025boltz2,
  title={Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  author={Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and others},
  journal={bioRxiv},
  year={2025}
}
```
