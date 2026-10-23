"""Prédire la gravité avec le modèle final, sans réentraînement."""
from pathlib import Path
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Coût conditionnel positif par contrat.')
    parser.add_argument('--input', type=Path, required=True, help='CSV au format du test initial.')
    parser.add_argument('--output', type=Path, required=True, help='CSV index,gravite_predite.')
    args = parser.parse_args()
    manifest = json.loads((root/'models/gravite_final.json').read_text())
    model_path = root/manifest['model_path']
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != manifest['model_sha256']:
        raise ValueError('Le modèle ne correspond pas à son manifeste.')
    frame = pd.read_csv(args.input, dtype={'code_postal': str})
    columns = manifest['raw_input_columns']
    missing = sorted((set(columns) | {'index'}) - set(frame.columns))
    if missing:
        raise ValueError('Colonnes manquantes : '+', '.join(missing))
    if frame['index'].isna().any() or not frame['index'].is_unique:
        raise ValueError('La colonne index doit être renseignée et unique.')
    model = joblib.load(model_path)
    prediction = model.predict(frame[columns])
    if len(prediction) != len(frame) or not np.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError('Les prédictions de gravité sont invalides.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'index': frame['index'].to_numpy(),
                  'gravite_predite': prediction}).to_csv(args.output, index=False)
    print(f'{len(frame):,} coûts conditionnels enregistrés dans {args.output}.')


if __name__ == '__main__':
    main()
