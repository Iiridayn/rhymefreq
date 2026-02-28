#!/usr/bin/env python3
"""
rhyme_families_basic.py
═══════════════════════
Builds a ranked list of the most rhymable word families in English.

Datasets used (all free/open):
  ┌─ CMUdict (phoneme transcriptions) ─────────────────────────────────────────┐
  │  Auto-downloaded by NLTK, or manually from:                                │
  │  https://github.com/cmusphinx/cmudict/raw/master/cmudict.dict              │
  │  ~134 k word forms with ARPAbet phonemes, American English.                │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ WordFreq (word frequencies) ───────────────────────────────────────────────┐
  │  pip install wordfreq                                                       │
  │  https://github.com/rspeer/wordfreq                                         │
  │  Zipf scale: 6 ≈ "the", 5 ≈ "love", 4 ≈ "rhyme",                         │
  │  3 ≈ borderline common, 2.5 ≈ uncommon-but-poetic, 2 ≈ rare.              │
  └────────────────────────────────────────────────────────────────────────────┘

Install:
  pip install nltk wordfreq
  python -m nltk.downloader cmudict

Variant pronunciations:
  CMUdict marks alternate pronunciations as word(2), word(3), etc.
  This script uses ALL pronunciations for each word, so a word like "either"
  (pronounced /iː/ or /aɪ/ depending on the speaker) will appear in BOTH
  phonetic rhyme families.  This gives breadth appropriate for poetry and
  lyrics, where variant pronunciations are legitimate creative choices.

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
import re
from collections import defaultdict

import nltk
from wordfreq import zipf_frequency

# ── Configuration ─────────────────────────────────────────────────────────────
ZIPF_CUTOFF     = 2.5   # Minimum Zipf score (log10-derived scale).
                         # 3.0 = ~1 per million words (common words only)
                         # 2.5 = ~0.3 per million (uncommon but poetically usable)
                         # 2.0 = rare; not generally recommended
                         # Raise toward 3.5 for a stricter, common-words-only output.
MIN_FAMILY_SIZE = 3      # Ignore families with fewer than this many members.
MAX_VARIANTS    = 6      # Max spelling variants shown in spelling_variants column.
OUTPUT_TSV      = "rhyme_families_basic.tsv"
# ─────────────────────────────────────────────────────────────────────────────

# Matches the "(2)", "(3)" suffixes CMUdict appends to alternate pronunciations.
_VARIANT_RE = re.compile(r'\(\d+\)$')


def strip_variant(word: str) -> str:
    """'either(2)' → 'either'"""
    return _VARIANT_RE.sub('', word)


def rhyme_unit(phonemes: list[str]) -> tuple[str, ...] | None:
    """
    Extract the rhyme unit from an ARPAbet phoneme list.

    Definition: all phonemes from the last *primary-stressed* vowel onward.
    Primary stress is marked by a trailing '1' in ARPAbet, e.g. 'AE1', 'OW1'.

    Examples:
      ['K', 'AE1', 'T']              → ('AE1', 'T')            cat
      ['N', 'AY1', 'T']              → ('AY1', 'T')            night
      ['R', 'AY1', 'T']              → ('AY1', 'T')            write (same family)
      ['R', 'IH0', 'T', 'ER1', 'N'] → ('ER1', 'N')            return
      ['IY1', 'DH', 'ER0']          → ('IY1', 'DH', 'ER0')    either (/iː/ pron.)
      ['AY1', 'DH', 'ER0']          → ('AY1', 'DH', 'ER0')    either (/aɪ/ pron.)

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

    Groups spelling variants within a phonetic rhyme family so we can show the
    most frequent example of each distinct spelling pattern.

    Examples (all in the /AY1 T/ family):
      'night'  → 'ight'
      'write'  → 'ite'
      'byte'   → 'yte'
      'fight'  → 'ight'   (same pattern as 'night')
    """
    last_v = -1
    for i, ch in enumerate(word.lower()):
        if ch in 'aeiou':
            last_v = i
    return word[last_v:].lower() if last_v >= 0 else word.lower()


def main():
    # ── 1. Load CMUdict ───────────────────────────────────────────────────────
    print("Loading CMUdict (downloading if needed)...")
    nltk.download('cmudict', quiet=True)
    # entries() returns (word_string, [phoneme, ...]) for every line in the dict,
    # including variant lines like ('either(2)', ['AY1', 'DH', 'ER0']).
    cmu_entries = nltk.corpus.cmudict.entries()
    print(f"  {len(cmu_entries):,} raw entries (including variants).")

    # ── 2. Collect all pronunciations per canonical word ──────────────────────
    print("Collecting all pronunciations per word...")
    word_pronunciations: dict[str, list[list[str]]] = defaultdict(list)
    for raw_word, phonemes in cmu_entries:
        canonical = strip_variant(raw_word).lower()
        word_pronunciations[canonical].append(phonemes)

    # ── 3. Filter by frequency; map each word to all its rhyme units ──────────
    print(f"Filtering (Zipf ≥ {ZIPF_CUTOFF}) and extracting rhyme units...")

    # families: rhyme_unit_tuple → {word: zipf_score}
    # Using a dict as the value deduplicates words that appear via multiple
    # pronunciations sharing the same rhyme unit.
    families: dict[tuple, dict[str, float]] = defaultdict(dict)

    kept, skipped_freq, skipped_stress = 0, 0, 0
    for word, pron_list in word_pronunciations.items():
        z = zipf_frequency(word, 'en')
        if z < ZIPF_CUTOFF:
            skipped_freq += 1
            continue

        units_seen: set[tuple] = set()
        for phonemes in pron_list:
            unit = rhyme_unit(phonemes)
            if unit is None:
                skipped_stress += 1
                continue
            if unit in units_seen:
                continue   # two variants happen to share the same rhyme unit
            units_seen.add(unit)
            families[unit][word] = z

        if units_seen:
            kept += 1

    print(f"  {kept:,} words retained  |  {skipped_freq:,} below Zipf  "
          f"|  {skipped_stress:,} stress-less skipped")
    print(f"  {len(families):,} distinct rhyme units found.")

    # ── 4. Build output rows ──────────────────────────────────────────────────
    print("Building output rows...")
    rows = []

    for unit, word_z_map in families.items():
        if len(word_z_map) < MIN_FAMILY_SIZE:
            continue

        members = sorted(word_z_map.items(), key=lambda x: x[1], reverse=True)
        rep_word, rep_zipf = members[0]

        # Best (highest-Zipf) word per orthographic ending
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

    # Sort: largest family first; representative Zipf as tiebreaker
    rows.sort(key=lambda r: (r['family_size'], r['rep_zipf']), reverse=True)

    # ── 5. Write TSV ──────────────────────────────────────────────────────────
    fields = ['rhyme_unit', 'family_size', 'representative',
              'rep_zipf', 'spelling_variants', 'all_words']

    with open(OUTPUT_TSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} families → {OUTPUT_TSV}")
    print(f"\nTop 40 most rhymable families:")
    print(f"{'Rank':<5} {'Rhyme Unit':<22} {'Size':>5}  {'Representative':<16} "
          f"{'Zipf':>5}  Spelling variants")
    print('─' * 100)
    for i, r in enumerate(rows[:40], 1):
        print(f"{i:<5} {r['rhyme_unit']:<22} {r['family_size']:>5}  "
              f"{r['representative']:<16} {r['rep_zipf']:>5}  "
              f"{r['spelling_variants'][:45]}")


if __name__ == '__main__':
    main()
