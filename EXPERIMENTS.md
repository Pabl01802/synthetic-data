# Experiment Log

Keep one section per meeting. Add the tables and plots you make from
`runs/*/results.json` and `test-results.json` (save the plots in `evidence/`),
link the pools' contact sheets, and explain what you see. A result that shows no
effect is still a result: say so and say why.

## Meeting 1 — Real-data baselines

- **Question:** How much do oversampling and ordinary augmentation help Egyptian
  Mau recall compared with the imbalanced data?
- **How `oversample()` works:** 55 = 3 × 15 + 10, so every one of the 15 photos
  is repeated 3 times and `rng.sample` picks 10 different photos for a fourth
  repeat. Each photo is added 3 or 4 times, and the order is shuffled with `rng`,
  so the seed reproduces the list. I checked this for seeds 0–4: 10 photos are
  added 4 times, 5 photos 3 times. The copies are not identical at training time,
  because `TRAIN_TRANSFORM` (crop, flip, mild colour jitter) still runs on every
  photo. So `oversample` is already a mildly augmented baseline.
- **How `augment()` works, and why these transforms:** it runs on the 55 added
  copies only, before `TRAIN_TRANSFORM`, and draws a new variant every epoch:
  - rotation up to ±15°, which the pipeline lacks; cats are photographed tilted;
  - a stronger crop (scale 0.6–1.0, ratio 0.8–1.25) for other framing and
    distance, since the pipeline only goes down to 0.75;
  - brightness and contrast ±0.3 and saturation ±0.2 for other lighting;
  - a light Gaussian blur (20% of the time) for soft or low-quality photos.

  Hue stays fixed and saturation changes less, because coat colour and spots are
  what identifies the breed. No vertical flips: upside-down cats are not in the
  data. [Examples](evidence/m1-augment-examples.jpg) (first column is the
  original): the cats stay recognisable, but rotation leaves black corners, and
  the crop can lose the head (row 2, column 4).
- **Commands:**

  ```bash
  uv run synth-train --name m1-real --method real
  uv run synth-train --name m1-oversample --method oversample
  uv run synth-train --name m1-augment --method augment
  uv run python evidence/m1_report.py val
  ```

- **Results on validation:** 5 seeds, 20 Egyptian Mau photos, so one photo is
  0.05 of recall.

  | Seed | Mau recall: real | oversample | augment | Macro-F1: real | oversample | augment |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 | 0.450 | 0.650 | 0.650 | 0.861 | 0.885 | 0.879 |
  | 1 | 0.500 | 0.600 | 0.650 | 0.851 | 0.878 | 0.872 |
  | 2 | 0.500 | 0.650 | 0.650 | 0.869 | 0.891 | 0.865 |
  | 3 | 0.400 | 0.750 | 0.800 | 0.835 | 0.912 | 0.898 |
  | 4 | 0.600 | 0.700 | 0.750 | 0.859 | 0.904 | 0.899 |
  | **Mean** | **0.490** | **0.670** | **0.700** | **0.855** | **0.894** | **0.882** |
  | SD | 0.074 | 0.057 | 0.071 | 0.013 | 0.014 | 0.015 |

  | Mean over seeds | real | oversample | augment |
  | --- | ---: | ---: | ---: |
  | Mean recall of the other 7 breeds | 0.914 | 0.929 | 0.911 |
  | Bengal recall | 0.840 | 0.820 | 0.720 |
  | Other breeds' photos called Egyptian Mau | 0.2 | 1.2 | 3.0 |
  | Egyptian Mau precision | 0.985 | 0.919 | 0.824 |

  ![Validation results](evidence/m1-val-results.png)

  Files: [per seed](evidence/m1-val-seeds.csv),
  [summary](evidence/m1-val-summary.csv),
  [paired differences](evidence/m1-val-differences.csv),
  [recall per breed](evidence/m1-val-breed-recall.csv),
  [per Mau photo](evidence/m1-val-mau-photos.csv).
- **What I see:**
  - **Balancing helps a lot.** Against `real`, recall rises in all 5 seeds with
    both methods: +0.18 with `oversample` (+0.10 to +0.35) and +0.21 with
    `augment` (+0.15 to +0.40), so about 4 more of the 20 photos.
  - **`augment` does not clearly beat `oversample` on recall.** The gap is +0.03,
    or 0.6 of a photo; `augment` is ahead in 3 seeds and tied in 2. This is too
    small to call with 20 photos and 5 seeds.
  - **`augment` costs more elsewhere.** Macro-F1 is lower than `oversample` in
    all 5 seeds (−0.012 on average), and so is the recall of the other breeds
    (−0.017). In seed 2 its macro-F1 is even below `real` (0.865 vs 0.869).
  - **The cost is Bengal.** Mau and Bengal are both spotted. `real` calls 5.0 Mau
    photos Bengal per seed; balancing cuts this to about 2, but moves the boundary
    the other way: Bengal photos called Mau go 0.2 → 1.2 → 3.0, and Bengal recall
    falls to 0.720 with `augment`. On validation, `oversample` has the best
    trade-off: nearly the same recall, and no measurable harm to the other breeds
    on average (+0.014), though Bengal itself drops slightly (−0.02).
  - **Some photos stay unsolved.** Of 20 photos, 8 are right in all 15 models, and
    3 in none ([sheet](evidence/m1-val-hard-mau.jpg): `Egyptian_Mau_15`, `_155`,
    `_131`; the bottom-right three are easy ones). By eye, the easy ones show a
    whole silver cat with clear spots, while the hard ones show a small cat in a
    room, a warm-coloured head close-up, or a curled-up bronze cat. My guess is
    that the 15 training photos do not cover these looks, and re-using those same
    photos cannot add them. I have not tested this; it is what synthetic images
    could try to fix.
  - **Limits.** Seeds change training only, not the 20 photos, so the spread
    between seeds understates the real uncertainty. Also, 560 training photos give
    9 batches per epoch and 505 give 8 (72 vs 64 steps), so `real` gets slightly
    less training; part of the gain could come from this, not only from balance.
  - **Expectation before testing:** both methods beat `real` on recall;
    `oversample` keeps the higher macro-F1. All three baselines go to test, as the
    README asks, so no choice is needed here.
- **Test results of the three baselines:** every seed of all three runs, on the
  240 test photos. There are 30 Egyptian Mau photos, so one photo is 0.033 of
  recall. These are the reference for the later meetings.

  ```bash
  uv run synth-test --run m1-real
  uv run synth-test --run m1-oversample
  uv run synth-test --run m1-augment
  uv run python evidence/m1_report.py test
  ```

  | Seed | Mau recall: real | oversample | augment | Macro-F1: real | oversample | augment |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 | 0.433 | 0.667 | 0.633 | 0.850 | 0.874 | 0.885 |
  | 1 | 0.500 | 0.633 | 0.600 | 0.853 | 0.873 | 0.855 |
  | 2 | 0.467 | 0.667 | 0.633 | 0.860 | 0.886 | 0.889 |
  | 3 | 0.433 | 0.700 | 0.667 | 0.850 | 0.891 | 0.882 |
  | 4 | 0.667 | 0.700 | 0.633 | 0.869 | 0.862 | 0.877 |
  | **Mean** | **0.500** | **0.673** | **0.633** | **0.857** | **0.877** | **0.878** |
  | SD | 0.097 | 0.028 | 0.024 | 0.008 | 0.011 | 0.013 |

  | Mean over seeds | real | oversample | augment |
  | --- | ---: | ---: | ---: |
  | Mean recall of the other 7 breeds | 0.914 | 0.908 | 0.915 |
  | Bengal recall | 0.933 | 0.900 | 0.887 |
  | Other breeds' photos called Egyptian Mau | 0.0 | 0.2 | 1.0 |
  | Egyptian Mau precision | 1.000 | 0.991 | 0.950 |

  ![Test results](evidence/m1-test-results.png)

  Files: [per seed](evidence/m1-test-seeds.csv),
  [summary](evidence/m1-test-summary.csv),
  [paired differences](evidence/m1-test-differences.csv),
  [recall per breed](evidence/m1-test-breed-recall.csv),
  [per Mau photo](evidence/m1-test-mau-photos.csv).

  - **The main result holds.** Balancing raises recall by about 0.13–0.17: +0.173
    with `oversample` (up in all 5 seeds) and +0.133 with `augment` (up in 4; in
    seed 4 it is below `real`, 0.633 vs 0.667). Macro-F1 rises by 0.021 with both,
    less than the +0.03 to +0.04 on validation.
  - **The order of `oversample` and `augment` on recall flipped.** On validation
    `augment` led by 0.03; on test `oversample` leads by 0.04, in all 5 seeds. This
    fits the validation note that the gap was too small to call. Ordinary
    augmentation shows no recall advantage over repeating the photos.
  - **My expectation was half right.** Both methods beat `real` on mean recall, but
    `oversample` did not keep the higher macro-F1: the two are tied (0.877 vs
    0.878).
  - **The Bengal harm is much smaller on test.** The direction is the same: Bengal
    recall 0.933 → 0.900 → 0.887, and Bengal photos called Mau 0 → 0.2 → 0.6. But
    the other breeds overall barely move (0.914, 0.908, 0.915). So the large
    validation drop was at least partly specific to those 20 Bengal photos.
  - **Mau is still the weakest breed.** The best baseline finds about 20 of 30
    photos, and most misses go to Bengal (4.2 per seed with `oversample`) and
    Russian Blue (4.2). Seven photos are missed by all 15 models and 10 are found
    by all 15, so, as on validation, a fixed group of photos stays out of reach of
    methods that re-use the same 15 photos.
  - **Reference for later meetings:** Egyptian Mau recall 0.673 and macro-F1 0.877
    (`oversample`). A synthetic method has to beat this, not only `real`.

## Meeting 2 — Text-to-image generation

- **Question:**
- **Setting changed in each pool (one at a time):**
- **Commands:**
- **Contact sheets:** (good and bad images, with the failure types you name)
- **Results on validation:** (table and plot)
- **What I see:**
- **Pool I take forward, and why:**
- **Its test result, compared with the baselines:**

## Meeting 3 — Generation from training photos

- **Question:** How close to the real photos should the synthetic images be?
- **img2img strength values and IP-Adapter scale values tried:**
- **Commands:**
- **Contact sheets:**
- **Results on validation:**
- **What I see:**
- **Best run of each mode, and its test result:**

## Meeting 4 — Editing training photos

- **Question:** Does the classifier need new cats or new backgrounds?
- **Background prompts and outpainting scales tried:**
- **Commands:**
- **Contact sheets:**
- **Results on validation:**
- **What I see:**
- **Best run of each mode, and its test result:**

## Final comparison

- **Runs chosen in Meetings 1–4:**
- **Results:** (your table and plot of their `test-results.json`)
- **Did the validation ranking hold on the test photos?**
- **Recommendation:** use, revise, or reject synthetic data for this task.
