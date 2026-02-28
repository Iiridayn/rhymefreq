#!/usr/bin/env python3
"""
rhyme_families_enhanced.py
══════════════════════════
Builds ranked rhyme family lists for English, classified by rhyme type:
  • masculine  — stress falls on the final syllable        ("return / concern")
  • feminine   — stress on penultimate, one trailing syllable ("lover / cover")
  • dactylic   — stress on antepenultimate or earlier      ("flattery / battery")

Datasets (all free/open):
  ┌─ CMUdict ───────────────────────────────────────────────────────────────────┐
  │  Auto-downloaded by NLTK, or directly:                                     │
  │  https://github.com/cmusphinx/cmudict/raw/master/cmudict.dict              │
  │  NLTK mirror: auto-fetched on first run by nltk.download('cmudict')        │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ WordFreq ──────────────────────────────────────────────────────────────────┐
  │  pip install wordfreq                                                       │
  │  https://github.com/rspeer/wordfreq                                         │
  │  Zipf scale: 6 ≈ "the", 5 ≈ "love", 4 ≈ "rhyme", 3 ≈ borderline common, │
  │  2 ≈ rare, <2 ≈ obscure.  We default to cutoff 3.0.                       │
  └────────────────────────────────────────────────────────────────────────────┘

Install:
  pip install nltk wordfreq
  python -m nltk.downloader cmudict

Outputs (one TSV per rhyme type + one combined):
  rhyme_families_masculine.tsv
  rhyme_families_feminine.tsv
  rhyme_families_dactylic.tsv
  rhyme_families_all.tsv          ← same rows, adds 'type' column, all together

Columns:
  type             — masculine | feminine | dactylic
  rhyme_unit       — ARPAbet phonemes from last stressed vowel onward
  syllables_after  — number of unstressed syllables after the stressed one
  family_size      — number of qualifying words in the family
  representative   — highest-frequency word in the family
  rep_zipf         — its Zipf score
  spelling_variants— most frequent word per distinct orthographic ending
  all_words        — all qualifying members, frequency-sorted
"""

import csv
from collections import defaultdict
from pathlib import Path

import nltk
from wordfreq import zipf_frequency

# ── Configuration ─────────────────────────────────────────────────────────────
ZIPF_CUTOFF     = 3.0   # Minimum Zipf score. Raise for stricter; lower for more.
MIN_FAMILY_SIZE = 3      # Skip families smaller than this.
MAX_VARIANTS    = 6      # Max spelling variants per family in output.
OUT_DIR         = Path(".")   # Output directory. Change as needed.
# ─────────────────────────────────────────────────────────────────────────────

VOWEL_SET = set('aeiou')


# ── Phoneme utilities ─────────────────────────────────────────────────────────

def is_vowel_ph(ph: str) -> bool:
    """ARPAbet vowels carry a stress digit (0, 1, or 2) as the final character."""
    return ph[-1] in '012'


def count_syllables_in(phonemes) -> int:
    """Count syllable nuclei (vowel phonemes) in a phoneme sequence."""
    return sum(1 for ph in phonemes if is_vowel_ph(ph))


def rhyme_unit_and_type(phonemes: list[str]) -> tuple | None:
    """
    Extract the rhyme unit and classify it.

    Rhyme unit = all phonemes from the last primary-stressed vowel onward.
    ('1' suffix in ARPAbet denotes primary stress: 'AE1', 'OW1', etc.)

    Classification by count of vowel phonemes in the rhyme unit:
      1 → masculine   (the stressed vowel IS the last syllable)
      2 → feminine    (one unstressed syllable follows the stressed one)
      3+ → dactylic   (two or more unstressed syllables follow)

    Returns (rhyme_unit_tuple, type_str, syllables_after_stress) or None.
    """
    last_stress_idx = None
    for i, ph in enumerate(phonemes):
        if ph.endswith('1'):
            last_stress_idx = i

    if last_stress_idx is None:
        return None

    unit = tuple(phonemes[last_stress_idx:])
    vowel_count = count_syllables_in(unit)   # includes the stressed vowel itself
    syllables_after = vowel_count - 1        # unstressed syllables after stress

    if vowel_count <= 1:
        rtype = 'masculine'
    elif vowel_count == 2:
        rtype = 'feminine'
    else:
        rtype = 'dactylic'

    return unit, rtype, syllables_after


# ── Orthographic utilities ────────────────────────────────────────────────────

def ortho_ending(word: str) -> str:
    """
    Return the orthographic rime: from the last vowel letter onward.

    Used to group spelling variants within a single phonetic rhyme family, e.g.:
      /AY1 T/ family: 'night' → 'ight', 'write' → 'ite', 'byte' → 'yte'

    We look for the last vowel in the word.  For words ending in silent 'e',
    the silent 'e' is included, correctly distinguishing 'ite' from 'it'.
    """
    last_v = -1
    for i, ch in enumerate(word.lower()):
        if ch in VOWEL_SET:
            last_v = i
    return word[last_v:].lower() if last_v >= 0 else word.lower()


# ── Family builder ────────────────────────────────────────────────────────────

def build_family_row(unit: tuple, members: list, rtype: str) -> dict:
    """
    Build one output row from a rhyme unit and its sorted member list.
    members: list of (word, zipf_score), sorted by score descending.
    """
    rep_word, rep_zipf = members[0]
    syllables_after = count_syllables_in(unit) - 1

    # Best word per orthographic ending
    by_ending: dict[str, tuple] = {}
    for word, z in members:
        ending = ortho_ending(word)
        if ending not in by_ending or z > by_ending[ending][1]:
            by_ending[ending] = (word, z)

    variants = sorted(by_ending.values(), key=lambda x: x[1], reverse=True)
    variant_str = ',  '.join(
        f"{w} ({z:.1f})" for w, z in variants[:MAX_VARIANTS]
    )

    return {
        'type':              rtype,
        'rhyme_unit':        ' '.join(unit),
        'syllables_after':   syllables_after,
        'family_size':       len(members),
        'representative':    rep_word,
        'rep_zipf':          round(rep_zipf, 2),
        'spelling_variants': variant_str,
        'all_words':         ', '.join(w for w, _ in members),
    }


def write_tsv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t',
                                extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows):,} rows → {path}")


def print_top(rows: list[dict], n: int = 20, label: str = "") -> None:
    if label:
        print(f"\n── Top {n} {label} families ──")
    print(f"{'Rhyme Unit':<26} {'Syl':>3} {'Size':>5}  {'Rep':<14} {'Zipf':>5}  Variants")
    print('─' * 95)
    for r in rows[:n]:
        print(f"{r['rhyme_unit']:<26} {r['syllables_after']:>3} {r['family_size']:>5}  "
              f"{r['representative']:<14} {r['rep_zipf']:>5}  "
              f"{r['spelling_variants'][:42]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load CMUdict ─────────────────────────────────────────────────────────
    print("Loading CMUdict (downloading if needed)...")
    nltk.download('cmudict', quiet=True)
    cmu_entries = nltk.corpus.cmudict.entries()
    print(f"  {len(cmu_entries):,} raw entries.")

    # 2. Deduplicate, filter, classify ────────────────────────────────────────
    print(f"Filtering (Zipf ≥ {ZIPF_CUTOFF}), extracting rhyme units...")

    seen = set()
    by_type: dict[str, dict[tuple, list]] = {
        'masculine': defaultdict(list),
        'feminine':  defaultdict(list),
        'dactylic':  defaultdict(list),
    }

    for word, phonemes in cmu_entries:
        if word in seen or '(' in word:
            continue
        seen.add(word)

        z = zipf_frequency(word, 'en')
        if z < ZIPF_CUTOFF:
            continue

        result = rhyme_unit_and_type(phonemes)
        if result is None:
            continue

        unit, rtype, _ = result
        by_type[rtype][unit].append((word, z))

    total_words = sum(
        sum(len(v) for v in d.values()) for d in by_type.values()
    )
    print(f"  {total_words:,} words retained across all types.")
    for rtype, d in by_type.items():
        print(f"    {rtype:10}: {sum(len(v) for v in d.values()):>6,} words "
              f"in {len(d):,} potential families")

    # 3. Build rows ───────────────────────────────────────────────────────────
    print("\nBuilding and ranking family rows...")

    all_rows   = []
    type_rows  = {}

    for rtype, families in by_type.items():
        rows = []
        for unit, members in families.items():
            if len(members) < MIN_FAMILY_SIZE:
                continue
            members.sort(key=lambda x: x[1], reverse=True)
            rows.append(build_family_row(unit, members, rtype))

        rows.sort(key=lambda r: (r['family_size'], r['rep_zipf']), reverse=True)
        type_rows[rtype] = rows
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r['family_size'], r['rep_zipf']), reverse=True)

    # 4. Write outputs ────────────────────────────────────────────────────────
    print("\nWriting output files...")

    base_fields = ['rhyme_unit', 'syllables_after', 'family_size',
                   'representative', 'rep_zipf', 'spelling_variants', 'all_words']
    all_fields  = ['type'] + base_fields

    for rtype, rows in type_rows.items():
        write_tsv(rows, OUT_DIR / f"rhyme_families_{rtype}.tsv", base_fields)

    write_tsv(all_rows, OUT_DIR / "rhyme_families_all.tsv", all_fields)

    # 5. Preview ──────────────────────────────────────────────────────────────
    for rtype, rows in type_rows.items():
        print_top(rows, n=15, label=rtype)

    print(f"\nSummary:")
    for rtype, rows in type_rows.items():
        print(f"  {rtype:10}: {len(rows):,} qualifying families")
    print(f"  {'combined':10}: {len(all_rows):,} qualifying families")


if __name__ == '__main__':
    main()
