#!/usr/bin/env python3
"""
rhyme_families_basic.py
═══════════════════════
Builds a ranked list of the most rhymable word families in English.

Datasets used (all free/open):
  ┌─ CMUdict (phoneme transcriptions) ─────────────────────────────────────────┐
  │  Auto-downloaded by NLTK, or manually from:                                │
  │  https://github.com/cmusphinx/cmudict/raw/master/cmudict.dict              │
  │  ~134 k word forms, ARPAbet phonemes, American English.                    │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ WordFreq (word frequencies) ───────────────────────────────────────────────┐
  │  pip install wordfreq                                                       │
  │  https://github.com/rspeer/wordfreq                                         │
  │  Compiled from many corpora; returns Zipf scores (0–8 scale).              │
  │  Zipf 6 ≈ "the", Zipf 3 ≈ ~1 per million words, Zipf 2 ≈ very rare.      │
  └────────────────────────────────────────────────────────────────────────────┘

Install:
  pip install nltk wordfreq
  python -m nltk.downloader cmudict

Output:
  rhyme_families_basic.tsv — tab-separated, one row per rhyme family, sorted
  by family size (most rhymable first).

Columns:
  rhyme_unit       — ARPAbet phonemes from last stressed vowel onward
  family_size      — number of qualifying words in the family
  representative   — highest-frequency word in the family
  rep_zipf         — its Zipf score
  spelling_variants— most frequent word per distinct orthographic ending
  all_words        — all qualifying members, frequency-sorted
"""

import csv
from collections import defaultdict

import nltk
from wordfreq import zipf_frequency

# ── Configuration ─────────────────────────────────────────────────────────────
ZIPF_CUTOFF     = 3.0   # Drop words below this Zipf score.
                         # 3.0 ≈ ~1 per million — excludes obscure words while
                         # keeping most words a literate adult would recognise.
                         # Lower to 2.5 to include more; raise to 3.5 to be strict.
MIN_FAMILY_SIZE = 3      # Ignore families with fewer than this many members.
MAX_VARIANTS    = 6      # Max spelling variants shown per family.
OUTPUT_TSV      = "rhyme_families_basic.tsv"
# ─────────────────────────────────────────────────────────────────────────────


def rhyme_unit(phonemes: list[str]) -> tuple[str, ...] | None:
    """
    Extract the rhyme unit from an ARPAbet phoneme list.

    Definition: all phonemes from the last *primary-stressed* vowel onward.
    Primary stress is indicated by the suffix '1' in ARPAbet, e.g. 'AE1'.

    Examples:
      ['K', 'AE1', 'T']           → ('AE1', 'T')          cat
      ['N', 'AY1', 'T']           → ('AY1', 'T')          night  ← same as...
      ['R', 'AY1', 'T']           → ('AY1', 'T')          write  ← ...these two
      ['R', 'IH0', 'T', 'ER1', 'N'] → ('ER1', 'N')       return

    Returns None for words with no primary stress (some abbreviations, etc.).
    """
    last_idx = None
    for i, ph in enumerate(phonemes):
        if ph.endswith('1'):
            last_idx = i
    return tuple(phonemes[last_idx:]) if last_idx is not None else None


def ortho_ending(word: str) -> str:
    """
    Orthographic rime: the substring from the last vowel letter onward.

    This groups spelling variants within a phonetic rhyme family so we can
    show the most frequent example of each distinct spelling pattern.

    Examples (all in the /AY1 T/ family):
      'night'  → 'ight'
      'write'  → 'ite'
      'byte'   → 'yte'
      'fight'  → 'ight'   (same pattern as 'night')
    """
    vowels = set('aeiou')
    last_v = -1
    for i, ch in enumerate(word.lower()):
        if ch in vowels:
            last_v = i
    return word[last_v:].lower() if last_v >= 0 else word.lower()


def main():
    # ── 1. Load CMUdict ───────────────────────────────────────────────────────
    print("Loading CMUdict (downloading if needed)...")
    nltk.download('cmudict', quiet=True)
    # entries: list of (word_string, [phoneme_string, ...])
    # Words with multiple pronunciations appear multiple times; we take the first.
    cmu_entries = nltk.corpus.cmudict.entries()
    print(f"  {len(cmu_entries):,} raw entries.")

    # ── 2. Deduplicate, filter by frequency, extract rhyme units ──────────────
    print(f"Filtering (Zipf ≥ {ZIPF_CUTOFF}) and extracting rhyme units...")

    seen_words = set()
    word_data  = []   # list of (word, zipf_score, rhyme_unit_tuple)

    for word, phonemes in cmu_entries:
        if word in seen_words:
            continue              # only use first/primary pronunciation
        seen_words.add(word)

        # CMUdict includes some entries with parenthesised variant markers like
        # "word(2)". Skip those — they're alternate pronunciations already
        # handled by the deduplication above.
        if '(' in word:
            continue

        z = zipf_frequency(word, 'en')
        if z < ZIPF_CUTOFF:
            continue

        unit = rhyme_unit(phonemes)
        if unit is None:
            continue

        word_data.append((word, z, unit))

    print(f"  {len(word_data):,} words retained.")

    # ── 3. Group into rhyme families ──────────────────────────────────────────
    print("Grouping into families...")
    families: dict[tuple, list] = defaultdict(list)
    for word, z, unit in word_data:
        families[unit].append((word, z))

    # Sort members by Zipf score descending within each family
    for unit in families:
        families[unit].sort(key=lambda x: x[1], reverse=True)

    # ── 4. Build output rows ──────────────────────────────────────────────────
    print("Building output rows...")
    rows = []

    for unit, members in families.items():
        if len(members) < MIN_FAMILY_SIZE:
            continue

        rep_word, rep_zipf = members[0]   # highest-frequency = representative

        # Spelling variants: best (highest Zipf) word per orthographic ending
        by_ending: dict[str, tuple] = {}
        for word, z in members:
            ending = ortho_ending(word)
            if ending not in by_ending or z > by_ending[ending][1]:
                by_ending[ending] = (word, z)

        variants = sorted(by_ending.values(), key=lambda x: x[1], reverse=True)
        variant_str = ',  '.join(
            f"{w} ({z:.1f})" for w, z in variants[:MAX_VARIANTS]
        )

        rows.append({
            'rhyme_unit':        ' '.join(unit),
            'family_size':       len(members),
            'representative':    rep_word,
            'rep_zipf':          round(rep_zipf, 2),
            'spelling_variants': variant_str,
            'all_words':         ', '.join(w for w, _ in members),
        })

    # Sort: largest family first; Zipf as tiebreaker
    rows.sort(key=lambda r: (r['family_size'], r['rep_zipf']), reverse=True)

    # ── 5. Write TSV ──────────────────────────────────────────────────────────
    fields = ['rhyme_unit', 'family_size', 'representative',
              'rep_zipf', 'spelling_variants', 'all_words']

    with open(OUTPUT_TSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} families → {OUTPUT_TSV}")
    print(f"\nTop 25 most rhymable families:")
    print(f"{'Rhyme Unit':<22} {'Size':>5}  {'Representative':<14} {'Zipf':>5}  Spelling variants")
    print('─' * 90)
    for r in rows[:25]:
        print(f"{r['rhyme_unit']:<22} {r['family_size']:>5}  "
              f"{r['representative']:<14} {r['rep_zipf']:>5}  "
              f"{r['spelling_variants'][:45]}")


if __name__ == '__main__':
    main()
