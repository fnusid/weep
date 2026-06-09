# Prompt TSE Experiment Log

Date: 2026-06-09

## Goal

Compare standard full-file target speaker extraction against prompt-conditioned target speaker extraction for DPCCN on LibriMix-style data, while staying as close as possible to the current `wesep` repo behavior.

## Reference Repo Behavior

Current reference files:

- [train_dpcnn.py](/home/sidcs/codebase/wesep/train_dpcnn.py:1)
- [dataset/dataloader.py](/home/sidcs/codebase/wesep/dataset/dataloader.py:1)

Important details from the current repo state:

- Test checkpoint used by `trainer.test(...)`:
  - `/home/sidcs/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt`
- Separate embedding checkpoint:
  - `/home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt`
- Test data path:
  - `mixture_test_mix_clean.csv`
- Test-time noise is added online in the dataloader:
  - noise probability: `0.8`
  - noise source: `wham_tt_noise_bins.json`
  - SNR sampling:
    - 40% from `[-5, 5]`
    - 40% from `[5, 15]`
    - 20% from `[15, 25]`
- Test-time target embedding selection in `train_dpcnn.py`:
  - use teacher embedding from clean target source
  - compute dual embeddings `e1`, `e2` from mixture
  - choose `e1` or `e2` by hard cosine comparison, not softmax blending

## Scripts Created / Updated

- [eval_libri05_08_full.py](/home/sidcs/codebase/wesep/eval_libri05_08_full.py:1)
  - DPCCN-only
  - evaluates untouched `Libriuni_05_08` test set
  - uses teacher-conditioned hard slot selection
  - includes progress bar

- [eval_prompt_test_set.py](/home/sidcs/codebase/wesep/eval_prompt_test_set.py:1)
  - DPCCN-only
  - compares `Mixture`, `Full`, and `Prompt`
  - evaluates suffix region after the prompt
  - uses teacher-conditioned hard slot selection
  - prints summary in table form

- [create_wesep_prompt_from_libri05_08_test.py](/home/sidcs/datasets/LibriMix/scripts/create_wesep_prompt_from_libri05_08_test.py:1)
  - builds 1-minute prompt benchmark from actual `Libri05_08` test examples
  - current final version uses:
    - fixed two-speaker pair per example
    - 15-second prompt
    - 60-second total length
    - dataloader-style online WHAM test noise

## Datasets We Tried

### 1. Synthetic 1-minute prompt set

Initial custom set built from concatenated clean LibriSpeech speaker audio with added noise.

Outcome:

- too different from original `Libri05_08` test distribution
- baseline SI-SDR was much easier than the original test set
- not appropriate for comparing against historical `Libri05_08` numbers

### 2. Stitched target-fixed / changing-interferer set

Second version built from actual `Libri05_08` examples but only fixed the target speaker.

Outcome:

- prompt conditioning failed badly
- interferer changed repeatedly across stitched segments
- prompt embedding from early audio did not remain stable/useful

This setup was rejected because it did not match the intended condition.

### 3. Final fixed-pair long-conversation set

Final prompt benchmark:

- path:
  - `/home/sidcs/datasets/LibriMix/LibriMix/wesep_prompt_test_1m_libri05_08style`
- source metadata:
  - `mixture_test_mix_clean.csv`
- construction:
  - stitch repeated occurrences of the same exact speaker pair from `Libri05_08`
  - choose one as target and keep the other fixed as interferer
  - prompt length: `15s`
  - total length: `60s`
  - online WHAM test noise added with current dataloader recipe

This is the final prompt benchmark used below.

## Most Important Results

### A. Untouched `Libri05_08` Full Evaluation

Command used:

```bash
/home/sidcs/miniconda3/envs/mtse/bin/python /home/sidcs/codebase/wesep/eval_libri05_08_full.py \
  --model dpccn \
  --tse-ckpt /home/sidcs/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt \
  --embedding-ckpt /home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt \
  --out-dir /home/sidcs/codebase/wesep/eval_runs/libri05_08_full_dpccn \
  --save-audio
```

Observed result:

| Condition | SI-SDR | SI-SDRi |
|-----------|--------|---------|
| Mixture   | -1.904 | -       |
| Full      | 10.604 | 12.507  |

Interpretation:

- this is in the same regime as the original remembered result
- the evaluation path is now much better aligned with the repo’s true test setup

### B. Prompt vs Full on Final Fixed-Pair Benchmark

Command used:

```bash
/home/sidcs/miniconda3/envs/mtse/bin/python /home/sidcs/codebase/wesep/eval_prompt_test_set.py \
  --model dpccn \
  --tse-ckpt /home/sidcs/model_ckpts/pDCCRN_2sp_dpccn/best-epoch=21-val_separation=0.000.ckpt \
  --embedding-ckpt /home/sidcs/model_ckpts/librispeech_asp_ft_wavlm_linear_dualemb_tr360/best-epoch=49-val_separation=0.000.ckpt \
  --metadata-csv /home/sidcs/datasets/LibriMix/LibriMix/wesep_prompt_test_1m_libri05_08style/metadata/prompt_test_metadata.csv \
  --out-dir /home/sidcs/codebase/wesep/eval_runs/prompt15_vs_full_dpccn_libri05_08style \
  --save-audio
```

Observed result:

| Condition | SI-SDR | SI-SDRi |
|-----------|--------|---------|
| Mixture   | -0.136 | -       |
| Full      | 2.251  | 2.387   |
| Prompt    | 3.655  | 3.791   |

Interpretation:

- prompt conditioning helps on this fixed-pair long-conversation benchmark
- prompt conditioning outperforms full-file conditioning in this setup
- likely reason:
  - the same two speakers persist through the whole clip
  - the first 15 seconds contain enough stable target-speaker evidence
  - prompt embedding remains useful for the later suffix

## Rejected / Incorrect Metric Setups

### Prompt scored on full 1-minute target while leaving first prompt region as raw mixture

Result:

- produced very negative prompt SI-SDR
- not meaningful

Reason:

- first prompt seconds still contain the interferer
- scoring against the clean target over the whole minute unfairly punishes the prompt setup

Fix:

- evaluate all three conditions on the same suffix region after the prompt

## Current Best Takeaways

1. The current DPCCN repo reproduces strong full-file TSE on untouched `Libri05_08` test:
   - `Mixture SI-SDR ≈ -1.9`
   - `Full SI-SDR ≈ 10.6`
   - `Full SI-SDRi ≈ 12.5`

2. Prompt TSE only becomes meaningful when the evaluation data actually matches the intended prompt condition.

3. On the final fixed-pair 15-second prompt benchmark:
   - `Prompt` outperforms `Full`
   - `Prompt SI-SDRi ≈ 3.79`
   - `Full SI-SDRi ≈ 2.39`

## Suggested Next Experiments

1. Repeat the fixed-pair prompt benchmark with prompt lengths:
   - `5s`
   - `15s`
   - `30s`

2. Increase the benchmark size from `10` examples to something like `50` or `100` examples.

3. Compare:
   - hard slot selection
   - softmax blending
   - online prompt embedding updates

4. Run the same prompt benchmark with any future non-DPCCN models after those models are added to this repo.
