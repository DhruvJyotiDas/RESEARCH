# CASMI26 working notes

## State (2026-11-22)
- Public frontier: best public notebooks are repackagings of the 0.500 stack. LB top 0.552; dense pack 0.51–0.53 (public = ~125 molecules, one rank-1 flip ~= 0.004).
- Our base: fork of the 0.500 stack (`dhruvjyoti/casmi26-base0500-repro`), submitted; score pending.
- Kaggle throughput is the bottleneck: 2 concurrent GPU sessions, and submission reruns appear to hold them for hours (kernels pushed before the two submissions ran immediately; after, they sat QUEUED for 3h+).

## Leak-free validation
Pseudo-tests are external spectra (MassBank-harmonized, GNPS-full extlib) of structures absent from train, run through the
whole stack on Kaggle with those structures removed from the external-library overlay.
- v1: 420 non-train molecules (36% in the COCONUT/LOTUS pool). v2: 293, mostly MULTIPLEX synthetics (dropped from the GPU
  plan: truth in PubChem window 17%). v3: 320 class-1 (train structures queried with external spectra, cos<0.98 to every
  train spectrum of the structure) + 32 non-train. External non-train supply is exhausted after v1/v2.
- Leakage traps found: PubChem depositor bit `src_b0` ("spectral library") is set for every pseudo truth by construction ->
  removed from all learned feature sets. External analog references inflate pseudo-test scores through same-source congener
  series (Eawag cyanopeptides) -> leave-source-out (LOSO) evaluation.

## PubChem-tier recall channel (channel.py)
All 105.9M tier structures within +-10 ppm of the neutral mass (adduct is given), Morgan-2 fingerprints, query-centric
analog propagation against the train library (modified cosine with mass shift), MS2 formula plausibility (explained
intensity by subformulae), formula-delta to analogs, PubChem documentation priors. ~250 s / 420 molecules on 4 cores.

| pseudo set | pop only | hand z(ap)+0.4 pop+NP+formula | learned (OOF, conservative feats) |
|---|---|---|---|
| v1 (non-train) | 0.131 | 0.243 | 0.365 (in-pool 0.679, top1 0.553, hit25 0.907; out-of-pool 0.189) |
| v3 (class 1) | 0.394 | 0.658 | 0.717 |

- Formula plausibility: true formula has the maximal explained fraction (ties incl.) in 79% of molecules.
- External references (extlib+MBH): non-LOSO v1 0.353 (spurious), LOSO 0.274 hand / 0.378 learned (+0.013) -> not used.

## Merger (make_merger_kernel.py)
Runs after the stack: features for stack candidates + channel top-K, LightGBM lambdarank trained on pseudo runs, score
z(-log final rank) + alpha * z(model), top-25 over the union, optional leader-keep margin. Validated locally end to end
(531 s on 400 molecules incl. ~140 s library build, peak RSS 5.2 GB). Waiting for pseudo v1/v3 stack outputs to train.

## Next
1. pseudo v1/v3 outputs -> `analyze_pseudo.py` (stage ranks) -> `prep_merger_data.py` (OOF channel scores) ->
   `train_merger.py` -> stack vs merged OOF MRR, by in_pool and class.
2. If merged > stack by a margin that survives CV: build merger kernel (`--channel_model`), submit, compare with 0.500.
