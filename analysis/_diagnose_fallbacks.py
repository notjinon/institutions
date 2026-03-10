"""One-off diagnostic: classify the 2,362 unmatched NIMSP candidates for 2004."""
import pandas as pd, sys, re, duckdb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.paths import UPPER_HOUSE_FILE, LOWER_HOUSE_FILE, DIME_PARQUET_FILE

def normalize_name(name):
    if pd.isna(name): return ''
    text = str(name).upper()
    text = re.sub(r'\([^)]*\)', ' ', text)
    text = re.sub(r'[^A-Z0-9\s]', ' ', text)
    tokens = [t for t in text.split() if t not in {'JR','SR','II','III','IV'}]
    return ' '.join(tokens)

def extract_last_name(name):
    if pd.isna(name): return ''
    text = str(name).upper().strip()
    if ',' in text:
        return normalize_name(text.split(',',1)[0])
    parts = normalize_name(text).split()
    return parts[-1] if parts else ''

year = 2004
frames = []
for path, house in [(UPPER_HOUSE_FILE(year),'U'),(LOWER_HOUSE_FILE(year),'L')]:
    try:
        df = pd.read_csv(path, low_memory=False); df['house'] = house; frames.append(df)
    except Exception as e: print(f'skip {path}: {e}')

nimsp = pd.concat(frames, ignore_index=True)
nimsp = nimsp[nimsp['General_Party'].str.upper().str.strip().isin(['DEMOCRATIC','REPUBLICAN'])]
nimsp['state']     = nimsp['Election_Jurisdiction'].str.upper().str.strip()
nimsp['name_norm'] = nimsp['Candidate'].map(normalize_name)
nimsp['last_norm'] = nimsp['Candidate'].map(extract_last_name)
nimsp['party']     = nimsp['General_Party'].str.upper().str.strip().map({'DEMOCRATIC':'DEM','REPUBLICAN':'REP'})

out = pd.read_csv('outputs/2004-analysis.csv', low_memory=False)
matched_keys = set(zip(out['nimsp_name'], out['state'], out['party']))
unmatched = nimsp[~nimsp.apply(
    lambda r: (r['name_norm'], r['state'], r['party']) in matched_keys, axis=1
)].copy()
print(f'NIMSP total (DEM+REP): {len(nimsp)}')
print(f'Matched in output:     {len(out)}')
print(f'Unmatched:             {len(unmatched)}')
print()

# Load DIME once
f = str(DIME_PARQUET_FILE(2004))
con = duckdb.connect()
dime = con.execute(
    'SELECT "recipient.name","recipient.state","recipient.party",seat '
    'FROM read_parquet(?) WHERE cycle=2004 AND "recipient.type"=\'CAND\'', [f]
).df()
con.close()
dime['name_norm'] = dime['recipient.name'].map(normalize_name)
dime['last_norm'] = dime['recipient.name'].map(extract_last_name)
print(f'DIME 2004 CAND rows: {len(dime):,}  unique names: {dime["name_norm"].nunique():,}')
print()

cats = {'exact_in_dime': [], 'last_matches': [], 'truly_absent': []}

for _, row in unmatched.iterrows():
    st = dime[dime['recipient.state'] == row['state']]
    if not st[st['name_norm'] == row['name_norm']].empty:
        cats['exact_in_dime'].append(row)
        continue
    sl = st[st['last_norm'] == row['last_norm']]
    if not sl.empty:
        cats['last_matches'].append({
            'nimsp_name': row['name_norm'],
            'dime_names':  '|'.join(sl['name_norm'].unique()[:4]),
            'dime_party':  '|'.join(sl['recipient.party'].dropna().unique()[:3]),
            'dime_seat':   '|'.join(sl['seat'].dropna().unique()[:3]),
            'state': row['state'], 'party': row['party'], 'house': row['house'],
        })
    else:
        cats['truly_absent'].append(row)

print(f"=== BREAKDOWN ===")
print(f"  Exact name in DIME (slipped through pool filter?): {len(cats['exact_in_dime'])}")
print(f"  Last name present, full name differs (fixable):    {len(cats['last_matches'])}")
print(f"  Truly absent from DIME (no donations):             {len(cats['truly_absent'])}")
print()

print("=== LAST-NAME-MATCH PAIRS (fixable — sample 60) ===")
mm = pd.DataFrame(cats['last_matches'])
if not mm.empty:
    print(mm.head(60).to_string(index=False))
print()

# Categorise the name-diff patterns
print("=== PATTERN ANALYSIS of last_matches ===")
if not mm.empty:
    # initial: NIMSP has e.g. "SIMITIAN S JOSEPH", DIME has "SIMITIAN JOE"
    initial_cases = mm[mm['nimsp_name'].str.split().str.len() >= 3]
    print(f"  NIMSP has 3+ tokens (middle initial/name issues): {len(initial_cases)}")
    two_tok = mm[mm['nimsp_name'].str.split().str.len() == 2]
    print(f"  NIMSP has 2 tokens (simple first+last):           {len(two_tok)}")
    one_tok = mm[mm['nimsp_name'].str.split().str.len() == 1]
    print(f"  NIMSP has 1 token (last only?):                   {len(one_tok)}")
