"""Préparation déterministe du référentiel Insee public, sans dépendance Excel.
Source : grille densité 2022, géographie 2021, population municipale 2018.
"""
from pathlib import Path
import json
import hashlib
import re
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_URL = 'https://www.insee.fr/fr/statistiques/fichier/6439600/grille_densite_7_niveaux_2021.xlsx'
PAGE_URL = 'https://www.insee.fr/fr/information/6439600'

def prepare_geography(root=ROOT):
    source = root / 'assets/grille_densite_7_niveaux_2021.xlsx'
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(source) as z:
        strings = [''.join(v.itertext()) for v in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rows = []
        for row in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//m:row', ns):
            values = {}
            for cell in row:
                column = re.sub(r'\d', '', cell.get('r', ''))
                value = cell.find('m:v', ns)
                value = value.text if value is not None else ''
                values[column] = strings[int(value)] if cell.get('t') == 's' and value else value
            if re.fullmatch(r'[0-9A-Z]{5}', values.get('A','')) and values.get('C','') in set('1234567'):
                rows.append({'code_commune':values['A'],'nom_commune':values['B'],
                             'densite7':int(values['C']), 'libelle_densite7':values['D'],
                             'population_2018':float(values['E'])})
    ref = pd.DataFrame(rows)
    assert ref.code_commune.is_unique and len(ref)>34000
    ref['densite3'] = ref.densite7.map({1:'urbain_dense',2:'urbain_intermediaire',
                         3:'urbain_intermediaire',4:'urbain_intermediaire',5:'rural',6:'rural',7:'rural'})
    path = root / 'assets/referentiel_communes_insee_2021.csv'
    ref.to_csv(path, index=False)
    meta = {'url':SOURCE_URL,'page':PAGE_URL,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'normalized_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'geography_year':2021,'population_year':2018,'published_in_file':'2023-05-17',
            'n_communes':len(ref),'claim_dates_unknown':True,
            'temporal_caveat':'Date des contrats inconnue; compatibilité temporelle du référentiel non garantie.'}
    (root/'assets/source_geographie.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    return ref

if __name__ == '__main__':
    print(prepare_geography().head().to_string(index=False))
