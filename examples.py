"""
Enhanced Boltz-2 Example Notebook
===================================
Demonstrates key features and use cases
"""

# Example 1: Basic Structure Prediction
# ======================================

from enhanced_boltz2 import *

config = EnhancedBoltzConfig(device="cuda", num_samples=5)
predictor = EnhancedBoltz2Predictor(config)

# Protein-ligand complex
protein = Molecule(id="A", molecule_type="protein", 
                   sequence="MLARALLLCAVLALSHTANP")
ligand = Molecule(id="LIG", molecule_type="ligand",
                  smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

request = PredictionRequest(molecules=[protein, ligand], predict_affinity=True)
result = predictor.predict(request)

print(f"Predicted IC50: {result['affinity']['ic50_um']:.3f} µM")
print(f"Confidence: {result['confidence']['overall']:.2f}")


# Example 2: Virtual Screening
# ==============================

pipeline = VirtualScreeningPipeline(predictor)

target_protein = "MLARALLLCAVLALSHTANP"
candidate_ligands = [
    "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
]

hits = pipeline.screen_ligands(
    protein_sequence=target_protein,
    ligand_smiles_list=candidate_ligands,
    affinity_threshold=-6.0
)

print(f"\nFound {len(hits)} hits:")
for i, hit in enumerate(hits, 1):
    print(f"{i}. IC50 = {hit['ic50_um']:.3f} µM (confidence: {hit['confidence']:.2f})")


# Example 3: DNA-Protein Complex
# ================================

dna = Molecule(id="D", molecule_type="dna", sequence="ATCGATCG")
protein = Molecule(id="P", molecule_type="protein", sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPIL")

request = PredictionRequest(
    molecules=[dna, protein],
    method_conditioning="xray",
    predict_affinity=False  # Structure only
)

result = predictor.predict(request)
print(f"Structure uncertainty: {result['confidence']['structure_uncertainty']:.3f}")


# Example 4: Antibody-Antigen Complex
# ====================================

antibody_heavy = Molecule(id="H", molecule_type="protein",
                          sequence="EVQLVESGGGLVQPGGSLRLSCAASGFTF")
antibody_light = Molecule(id="L", molecule_type="protein",
                          sequence="DIQMTQSPSSLSASVGDRVTITC")
antigen = Molecule(id="A", molecule_type="protein",
                   sequence="MKTAYIAKQRQISFVK")

request = PredictionRequest(
    molecules=[antibody_heavy, antibody_light, antigen],
    predict_affinity=True
)

result = predictor.predict(request)
print(f"Binding affinity: {result['affinity']['value']:.2f}")


# Example 5: Batch Processing
# =============================

# Create multiple requests
requests = []
for i in range(10):
    smiles = candidate_ligands[i % len(candidate_ligands)]
    molecules = [
        Molecule(id="A", molecule_type="protein", sequence=target_protein),
        Molecule(id="LIG", molecule_type="ligand", smiles=smiles)
    ]
    requests.append(PredictionRequest(molecules=molecules, predict_affinity=True))

# Batch predict
results = predictor.batch_predict(requests)

print(f"\nProcessed {len(results)} complexes")
mean_affinity = sum(r['affinity']['value'] for r in results) / len(results)
print(f"Mean affinity: {mean_affinity:.2f}")
