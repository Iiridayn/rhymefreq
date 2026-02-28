# English Rhyme Families

A data pipeline that builds **ranked lists of the most rhymable word families in English**, derived entirely from open, freely available datasets. Useful for poets, lyricists, game designers, NLP researchers, and anyone else who needs structured rhyme data programmatically.

---

## Background

This project was designed in collaboration with an AI assistant. The full design conversation — covering the theory of rhyme, phoneme-based detection, dataset selection, and stemming considerations — is available here:

> 🔗 **[Design conversation](https://claude.ai/share/108d6033-f727-45dd-a079-e2024fabc846)**

---

## How it works

English spelling is notoriously unreliable for detecting rhyme (`night`, `write`, `byte` all rhyme despite entirely different letter patterns). This pipeline therefore works at the **phoneme level**, not the spelling level.

### Data sources

| Dataset | What it provides | License |
|---|---|---|
| [CMUdict](https://github.com/cmusphinx/cmudict) | ARPAbet phoneme transcriptions for ~134k English words | BSD-compatible |
| [WordFreq](https://github.com/rspeer/wordfreq) | Zipf-scale word frequencies compiled from many corpora | MIT |

Both are auto-fetched on first run — no manual download needed.

### Pipeline

1. **Load CMUdict** — including all variant pronunciations (`word(2)`, `word(3)`, etc.)
2. **Collect all pronunciations per word** — a word with two valid pronunciations contributes to both rhyme families
3. **Filter by frequency** — words below the Zipf cutoff (default: **2.5**) are excluded. This retains uncommon but poetically usable words while dropping genuine obscurities
4. **Extract rhyme units** — all ARPAbet phonemes from the last primary-stressed vowel (marked `*1`) onward
5. **Group words by rhyme unit** — these are the rhyme families
6. **Characterise each family** — representative word (highest frequency), family size, and the most frequent word per distinct orthographic spelling ending
7. **Rank and export** — largest families first, ties broken by representative word frequency

### Zipf scale reference

| Zipf score | Approx. frequency | Examples |
|---|---|---|
| 6 | ~1 per 1,000 words | the, is, and |
| 5 | ~1 per 10,000 | love, time, make |
| 4 | ~1 per 100,000 | rhyme, comet, flicker |
| 3.0 | ~1 per million | borderline common |
| **2.5** | ~0.3 per million | **default cutoff** — poetically usable |
| 2.0 | ~0.1 per million | rare; not recommended |

### Variant pronunciations

CMUdict annotates alternate pronunciations as `word(2)`, `word(3)`, etc. Previous tools typically discard these; this pipeline uses all of them. A word like `either` (pronounced `/iː/` by some speakers and `/aɪ/` by others) will appear in **both** rhyme families, reflecting real creative usage in poetry and lyrics.

---

## Scripts

### `rhyme_families_basic.py`

Produces a single TSV of masculine rhyme families (stress on final syllable), sorted by family size.

**Install:**
```bash
pip install nltk wordfreq
python -m nltk.downloader cmudict
```

**Run:**
```bash
python rhyme_families_basic.py
```

**Output:** `rhyme_families_basic.tsv`

| Column | Description |
|---|---|
| `rhyme_unit` | ARPAbet phonemes from last stressed vowel onward |
| `family_size` | Number of qualifying words in the family |
| `representative` | Highest-frequency word in the family |
| `rep_zipf` | Zipf score of the representative word |
| `spelling_variants` | Most frequent word per distinct orthographic ending |
| `all_words` | All qualifying members, frequency-sorted |

---

### `rhyme_families_enhanced.py`

Extends the basic script to classify rhyme families by type:

| Type | Definition | Example |
|---|---|---|
| **Masculine** | Stress on final syllable | *return / concern* |
| **Feminine** | Stress on penultimate; one unstressed syllable follows | *lover / cover* |
| **Dactylic** | Two or more unstressed syllables follow the stress | *flattery / battery* |

**Run:**
```bash
python rhyme_families_enhanced.py
```

**Outputs:**
- `rhyme_families_masculine.tsv`
- `rhyme_families_feminine.tsv`
- `rhyme_families_dactylic.tsv`
- `rhyme_families_all.tsv` (all types combined, with a `type` column)

---

## Configuration

Both scripts share the same configuration block near the top:

```python
ZIPF_CUTOFF     = 2.5   # Raise to ~3.5 for common words only; lower to 2.0 for more breadth
MIN_FAMILY_SIZE = 3      # Minimum members for a family to appear in output
MAX_VARIANTS    = 6      # Max spelling variants shown per family
```

---

## Top 40 most rhymable families (basic script, Zipf ≥ 2.5)

Rank  Rhyme Unit              Size  Representative    Zipf  Spelling variants
────────────────────────────────────────────────────────────────────────────────────────────────────
1     EY1 SH AH0 N             575  information       5.43  information (5.4),  corp (4.3),  corp. (4.3),
2     IY1                      223  be                6.79  be (6.8),  d (5.5),  d. (5.5),  t (5.4),  t. 
3     EY1 SH AH0 N Z           201  patients          4.91  patients (4.9),  operations (4.9),  operation
4     UW1                      193  to                7.43  to (7.4),  you (7.0),  new (6.2),  through (5
5     EY1                      193  a                 7.36  a (7.4),  a. (7.4),  they (6.5),  way (6.0), 
6     OW1                      140  so                6.52  so (6.5),  know (6.1),  though (5.5),  oh (5.
7     IY1 Z                    132  these             6.04  these (6.0),  he's (5.6),  d.'s (5.3),  t.'s 
8     EY1 N                    130  again              5.7  again (5.7),  campaign (5.0),  plane (4.7),  
9     AY1                      127  i                 7.09  i (7.1),  i. (7.1),  by (6.7),  my (6.6),  wh
10    IY1 N                    121  between           5.77  between (5.8),  mean (5.5),  scene (5.0),  jo
11    EH1 L                    111  well              6.03  well (6.0),  l (5.2),  l. (5.2),  hotel (5.0)
12    AA1 N                     92  on                6.91  on (6.9),  john (5.4),  iran (4.6),  antoine 
13    EY1 Z                     91  a.'s              5.85  a.'s (5.8),  days (5.6),  days' (5.6),  j.'s 
14    EH1 S                     89  us                6.04  us (6.0),  's (5.9),  s (5.9),  s. (5.9),  ye
15    EY1 D                     89  made              5.92  made (5.9),  played (5.2),  paid (5.1),  they
16    EH1 R                     88  their             6.33  their (6.3),  there (6.3),  wear (4.9),  pray
17    AO1 R                     87  for               7.01  for (7.0),  your (6.5),  more (6.4),  war (5.
18    UW1 Z                     85  use               5.81  use (5.8),  news (5.4),  news' (5.4),  u.s (5
19    AE1 N                     82  an                6.53  an (6.5),  jan. (4.5),  ann (4.3),  anne (4.3
20    EH1 T                     82  get               6.28  get (6.3),  debt (4.7),  threat (4.6),  cigar
21    IH1 SH AH0 N              82  position          5.21  position (5.2),  physician (4.2)
22    IH1 R                     76  year              5.96  year (6.0),  we're (5.4),  career (5.1),  ami
23    EH1 S T                   75  best              5.84  best (5.8),  expressed (4.5),  breast (4.3), 
24    AY1 Z                     72  i.'s              5.84  i.'s (5.8),  guys (5.4),  guys' (5.4),  eyes 
25    AY1 D                     70  side               5.5  side (5.5),  i'd (5.4),  died (5.2),  eid (3.
26    OW1 Z                     69  those              5.9  those (5.9),  shows (5.2),  shows' (5.2),  go
27    EY1 T                     69  great             5.88  great (5.9),  state (5.8),  wait (5.3),  m-8 
28    AA1 R                     65  are               6.74  are (6.7),  our (6.1),  far (5.5),  r (5.3), 
29    AY1 T                     65  right             5.96  right (6.0),  white (5.5),  beit (3.0),  indi
30    IY1 N AH0                 65  arena             4.33  arena (4.3)
31    IH1 N                     64  in                7.27  in (7.3),  in. (7.3),  when (6.4),  inn (4.1)
32    OW1 N                     63  don't              6.2  don't (6.2),  own (5.7),  phone (5.3),  loan 
33    IY1 D                     63  need              5.97  need (6.0),  read (5.5),  he'd (4.6),  reid (
34    IY1 L                     63  feel              5.67  feel (5.7),  real (5.6),  we'll (4.9),  neil 
35    EY1 L                     61  sale              4.99  sale (5.0),  mail (4.7),  they'll (4.7),  gae
36    EY1 N Z                   61  remains            4.8  remains (4.8),  campaigns (4.2),  planes (4.2
37    IH1 L                     60  will              6.45  will (6.5),  until (5.6),  we'll (4.9),  sevi
38    AA1                       59  law               5.46  law (5.5),  la (5.0),  ah (4.6),  ahh (3.9), 
39    AE1 N AH0                 59  santa             4.49  santa (4.5),  piano (4.3),  hannah (4.0),  sa
40    AW1                       57  how               6.24  how (6.2),  thou (4.2),  mao (3.6),  howe (3.

---

## Notes and caveats

- **American English only.** CMUdict reflects General American pronunciation. British variants (e.g. the non-rhotic vowel in `car`) are not represented.
- **Primary pronunciation only per variant group.** If `word(2)` and `word(3)` happen to produce the same rhyme unit, it is counted only once.
- **No lemmatisation.** The pipeline operates on surface forms as they appear in CMUdict. Highly inflected forms (`running`, `runs`) are present but typically fall below the Zipf cutoff, since WordFreq's base frequencies are lemma-weighted. If you need strict lemma-only output, add a spaCy lemmatisation step before the CMUdict lookup.
- **Slant/near rhyme is not included.** Only perfect phonetic rhyme (identical rhyme unit) is detected. Assonance and consonance are out of scope.

---

## License

Scripts: MIT. Dataset terms: see [CMUdict license](https://github.com/cmusphinx/cmudict/blob/master/LICENSE) and [WordFreq license](https://github.com/rspeer/wordfreq/blob/master/LICENSE.txt).
