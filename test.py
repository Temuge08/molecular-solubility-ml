# %%
"""
Demo: using canonicalize() to standardize SMILES strings with RDKit.
"""

from rdkit import Chem


def canonicalize(mol):
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
    except Exception:
        return None


if __name__ == "__main__":
    # --- 1. Different SMILES strings for the SAME molecule (ethanol) ---
    # canonicalize() should collapse them to one identical string.
    ethanol_variants = ["CCO", "OCC", "C(O)C", 'OCC(O)C(O)C(O)C(O)CO']

    ethanol_variants_mol = []
    print("=== Canonicalization: same molecule, different input SMILES ===")
    for smi in ethanol_variants:
        mol = Chem.MolFromSmiles(smi)          # string -> Mol object
        # canon = canonicalize(mol)               # Mol object -> canonical SMILES
        ethanol_variants_mol.append(mol)
    print("ethanol_variants_mol:", ethanol_variants_mol)
        
    
    # for mol in ethanol_variants_mol:
    #     canoni = canonicalize(mol)
    #     print('Before', mol)
    #     print('After', canoni)
    

    # --- 2. Stereochemistry is preserved (isomericSmiles=True) ---
    # (R)- and (S)-alanine are enantiomers: same connectivity, different
    # spatial arrangement, so they must NOT canonicalize to the same string.
    # r_alanine = "C[C@@H](N)C(=O)O"
    # s_alanine = "C[C@H](N)C(=O)O"

    # print("\n=== Stereochemistry is preserved ===")
    # for label, smi in [("R-alanine", r_alanine), ("S-alanine", s_alanine)]:
    #     mol = Chem.MolFromSmiles(smi)
    #     print(f"  {label}: {smi} -> {canonicalize(mol)}")

    # # --- 3. Handling invalid / unparsable input gracefully ---
    # print("\n=== Invalid input handling ===")
    # bad_smiles = "not_a_real_molecule("
    # bad_mol = Chem.MolFromSmiles(bad_smiles)   # RDKit returns None on failure
    # print(f"  Chem.MolFromSmiles('{bad_smiles}') -> {bad_mol}")
    # print(f"  canonicalize(bad_mol) -> {canonicalize(bad_mol)}")

    # # --- 4. Practical use case: deduplicating a list of molecules ---
    # raw_list = ["CCO", "OCC", "c1ccccc1", "C1=CC=CC=C1", "CCO"]
    # print("\n=== Deduplicating molecules via canonical SMILES ===")
    # seen = set()
    # unique = []
    # for smi in raw_list:
    #     canon = canonicalize(Chem.MolFromSmiles(smi))
    #     if canon and canon not in seen:
    #         seen.add(canon)
    #         unique.append(canon)
    # print(f"  raw ({len(raw_list)}):    {raw_list}")
    # print(f"  unique ({len(unique)}): {unique}")

# %%



