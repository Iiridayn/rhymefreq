#!/usr/bin/env python3
"""
rhyme_families_enhanced.py
══════════════════════════
Builds ranked rhyme family lists for English, classified by rhyme type:
  • masculine  — stress on the final syllable            ("return / concern")
  • feminine   — stress on penultimate; one trailing syl  ("lover / cover")
  • dactylic   — two or more unstressed syllables follow  ("flattery / battery")

Datasets (all free/open):
  ┌─ CMUdict ───────────────────────────────────────────────────────────────────┐
  │  Auto-downloaded by NLTK on first run, or directly from:                   │
  │  https://github.com/cmusphinx/cmudict/raw/master/cmudict.dict              │
  └────────────────────────────────────────────────────────────────────────────┘
  ┌─ WordFreq ──────────────────────────────────────────────────────────────────┐
  │  pip install wordfreq       https://github.com/rspeer/wordfreq             │
  │  Zipf scale: 6 ≈ "the", 5 ≈ "love", 4 ≈ "rhyme",                         │
  │  3 ≈ borderline common, 2.5 ≈ uncommon-but-poetic.                        │
  └────────────────────────────────────────────────────────────────────────────┘

Install:
  pip install nltk wordfreq
  python -m nltk.downloader cmudict

Variant pronunciations:
  CMUdict marks alternate pronunciations as word(2), word(3), etc.
  This script uses ALL pronunciations for each word and maps each word into
  every rhyme family its variants belong to.  A word may therefore appear in
  multiple families (e.g. "either" in both the /IY/-ending and /AY/-ending
  families) and may even appear in families of different types (e.g. a word
  whose primary pronunciation is masculine but whose variant is feminine).
  This breadth is intentional for poetry / lyrics use.

Outputs:
  rhyme_families_masculine.tsv
  rhyme_families_feminine.tsv
  rhyme_families_dactylic.tsv
  rhyme_families_all.tsv          ← all families with a 'type' column added

Columns:
  type             — masculine | feminine | dactylic
  rhyme_unit       — ARPAbet phonemes from last stressed vowel onward
  syllables_after  — count of unstressed syllables following the stressed one
  family_size      — number of qualifying words in the family
  representative   — highest-frequency word in the family
  rep_zipf         — its Zipf score
  spelling_variants— most frequent word per distinct orthographic ending
  all_words        — all qualifying members, frequency-sorted
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import nltk
from wordfreq import zipf_frequency

# ── Configuration ─────────────────────────────────────────────────────────────
ZIPF_CUTOFF     = 2.5   # Minimum Zipf score.
                         # 3.0 = common words only (~1/million)
                         # 2.5 = uncommon but poetically acceptable (~0.3/million)
MIN_FAMILY_SIZE = 3      # Skip families with fewer members than this.
MAX_VARIANTS    = 6      # Max spelling variants shown per family.
OUT_DIR         = Path(".")
# ─────────────────────────────────────────────────────────────────────────────

VOWEL_SET   = set('aeiou')
_VARIANT_RE = re.compile(r'\(\d+\)$')


# ── Phoneme utilities ─────────────────────────────────────────────────────────

def strip_variant(word: str) -> str:
    return _VARIANT_RE.sub('', word)


def is_vowel_ph(ph: str) -> bool:
    """ARPAbet vowel phonemes end with a stress digit (0, 1, or 2)."""
    return ph[-1] in '012'


def count_vowels(phonemes) -> int:
    return sum(1 for ph in phonemes if is_vowel_ph(ph))


def rhyme_unit_and_type(phonemes: list[str]) -> tuple | None:
    """
    Extract rhyme unit and classify by type.

    Rhyme unit = all phonemes from the last primary-stressed vowel ('*1') onward.

    Type classification by vowel phoneme count in the rhyme unit:
      1 vowel  → masculine   (stressed vowel is the last syllable nucleus)
      2 vowels → feminine    (one unstressed syllable trails the stressed one)
      3+ vowels→ dactylic    (two or more unstressed syllables trail)

    Returns (rhyme_unit_tuple, type_str, syllables_after) or None.
    """
    last_stress_idx = None
    for i, ph in enumerate(phonemes):
        if ph.endswith('1'):
            last_stress_idx = i

    if last_stress_idx is None:
        return None

    unit = tuple(phonemes[last_stress_idx:])
    vowel_count    = count_vowels(unit)
    syllables_after = vowel_count - 1    # subtract the stressed vowel itself

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
    Orthographic rime: from the last vowel letter onward.

    Used to surface spelling variants within one phonetic family.
    E.g. in the /AY1 T/ family: 'night'→'ight', 'write'→'ite', 'byte'→'yte'.
    Silent final 'e' is included, correctly distinguishing 'ite' from 'it'.
    """
    last_v = -1
    for i, ch in enumerate(word.lower()):
        if ch in VOWEL_SET:
            last_v = i
    return word[last_v:].lower() if last_v >= 0 else word.lower()


# ── Family builder ────────────────────────────────────────────────────────────

def build_family_row(unit: tuple, members: list, rtype: str) -> dict:
    """
    Build one output row for a rhyme family.
    members: [(word, zipf_score), ...] sorted by score descending.
    """
    rep_word, rep_zipf = members[0]
    syllables_after = count_vowels(unit) - 1

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
    print(f"{'Rank':<5} {'Rhyme Unit':<28} {'Syl':>3} {'Size':>5}  "
          f"{'Rep':<16} {'Zipf':>5}  Variants")
    print('─' * 100)
    for i, r in enumerate(rows[:n], 1):
        print(f"{i:<5} {r['rhyme_unit']:<28} {r['syllables_after']:>3} "
              f"{r['family_size']:>5}  {r['representative']:<16} "
              f"{r['rep_zipf']:>5}  {r['spelling_variants'][:40]}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load CMUdict ─────────────────────────────────────────────────────────
    print("Loading CMUdict (downloading if needed)...")
    nltk.download('cmudict', quiet=True)
    cmu_entries = nltk.corpus.cmudict.entries()
    print(f"  {len(cmu_entries):,} raw entries (including variants).")

    # 2. Collect all pronunciations per canonical word ─────────────────────────
    print("Collecting all pronunciations per word...")
    word_pronunciations: dict[str, list[list[str]]] = defaultdict(list)
    for raw_word, phonemes in cmu_entries:
        canonical = strip_variant(raw_word).lower()
        word_pronunciations[canonical].append(phonemes)

    # 3. Filter by frequency; classify each (word, pronunciation) pair ─────────
    print(f"Filtering (Zipf ≥ {ZIPF_CUTOFF}), classifying rhyme types...")

    # by_type[rtype][unit] = {word: zipf_score}
    by_type: dict[str, dict[tuple, dict[str, float]]] = {
        'masculine': defaultdict(dict),
        'feminine':  defaultdict(dict),
        'dactylic':  defaultdict(dict),
    }

    kept, skipped_freq, skipped_stress = 0, 0, 0
    for word, pron_list in word_pronunciations.items():
        z = zipf_frequency(word, 'en')
        if z < ZIPF_CUTOFF:
            skipped_freq += 1
            continue

        # Track (unit, type) pairs seen for this word to avoid double-counting
        # when two variant pronunciations map to the same rhyme unit + type.
        seen_unit_type: set[tuple] = set()
        placed = False
        for phonemes in pron_list:
            result = rhyme_unit_and_type(phonemes)
            if result is None:
                skipped_stress += 1
                continue
            unit, rtype, _ = result
            key = (unit, rtype)
            if key in seen_unit_type:
                continue
            seen_unit_type.add(key)
            by_type[rtype][unit][word] = z
            placed = True

        if placed:
            kept += 1

    print(f"  {kept:,} words retained  |  {skipped_freq:,} below Zipf  "
          f"|  {skipped_stress:,} stress-less skipped")
    for rtype, d in by_type.items():
        total_members = sum(len(v) for v in d.values())
        print(f"    {rtype:10}: {total_members:>6,} word-placements "
              f"across {len(d):,} potential families")

    # 4. Build rows ────────────────────────────────────────────────────────────
    print("\nBuilding and ranking family rows...")

    all_rows:  list[dict] = []
    type_rows: dict[str, list[dict]] = {}

    for rtype, families in by_type.items():
        rows = []
        for unit, word_z_map in families.items():
            if len(word_z_map) < MIN_FAMILY_SIZE:
                continue
            members = sorted(word_z_map.items(), key=lambda x: x[1], reverse=True)
            rows.append(build_family_row(unit, members, rtype))

        rows.sort(key=lambda r: (r['family_size'], r['rep_zipf']), reverse=True)
        type_rows[rtype] = rows
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r['family_size'], r['rep_zipf']), reverse=True)

    # 5. Write outputs ─────────────────────────────────────────────────────────
    print("\nWriting output files...")
    base_fields = ['rhyme_unit', 'syllables_after', 'family_size',
                   'representative', 'rep_zipf', 'spelling_variants', 'all_words']
    all_fields  = ['type'] + base_fields

    for rtype, rows in type_rows.items():
        write_tsv(rows, OUT_DIR / f"rhyme_families_{rtype}.tsv", base_fields)
    write_tsv(all_rows, OUT_DIR / "rhyme_families_all.tsv", all_fields)

    # 6. Preview ───────────────────────────────────────────────────────────────
    for rtype, rows in type_rows.items():
        print_top(rows, n=15, label=rtype)

    print(f"\nSummary:")
    for rtype, rows in type_rows.items():
        print(f"  {rtype:10}: {len(rows):,} qualifying families "
              f"(≥{MIN_FAMILY_SIZE} members, Zipf ≥{ZIPF_CUTOFF})")
    print(f"  {'combined':10}: {len(all_rows):,} qualifying families total")


if __name__ == '__main__':
    main()
