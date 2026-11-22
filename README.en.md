# faster-whisper ASR Cost Benchmark (EC2 GPU Instance Cost-Efficiency)

[简体中文](README.md) | **English**

Runs offline batch transcription with **faster-whisper large-v3** on different AWS EC2 GPU
instances (**G5 / G6 / G6e / G7 / G7e**). It fully automatically pushes the GPU **VRAM to its
limit** (filling ~90% or blowing it up with an OOM), and on that basis measures **RTF
(real-time factor)** and **inference time**. Combined with the **On-Demand price**, it then
computes the **cost per audio hour**, generates a cost analysis report for each instance, and
can aggregate the results into a cross-instance cost-efficiency comparison.

Core formula:

```
Cost per audio hour ($/audio-hour) = Instance On-Demand hourly price ($/hour) ÷ RTF
```

RTF means "how many hours of audio one hour of machine wall-clock time can transcribe". So an
instance with a higher hourly price but faster speed ends up with a lower unit cost, as long as
its RTF gain exceeds its price increase — this project lets the data decide directly which
instance is the best deal.

> **Why "fill the VRAM"**: `large-v3` decodes in 30s windows, and `BatchedInferencePipeline`
> first uses VAD to split the audio into speech segments, then processes them in parallel
> batches. **When the audio is too short, it yields fewer segments than batch_size, so large
> batches can't be filled**, and VRAM usage and RTF "falsely plateau" long before the hardware
> limit, **systematically underestimating stronger GPUs**. This project adaptively grows the
> audio to make sure the measured throughput reflects a truly saturated GPU.

---

## Quick Start (on the target instance)

```bash
git clone <YOUR_REPO_URL>.git
cd faster-whisper-cost-benchmark

# 1) One-step environment setup (ffmpeg + venv + torch cu128 + latest faster-whisper)
bash setup.sh
source .venv/bin/activate

# 2) Run the benchmark: fully automatic. Generates audio → fills VRAM → computes RTF/time/cost
python benchmark.py
```

`benchmark.py` **generates the test audio automatically** (20× the base sample by default), so
there is no need to run `prepare_audio.py` manually. When the run finishes, reports are written
to `results/`:
- `results/<instance>.json`: complete machine-readable results (including VRAM utilization and
  saturation status)
- `results/<instance>.md`: the cost analysis report for this instance

After repeating this on every target instance, collect each instance's
`results/<instance>.json` in one place:

```bash
python aggregate_report.py --results-dir results
# => results/COST-COMPARISON.md  cross-instance cost-efficiency comparison table (with a VRAM utilization column)
```

---

## How It Works

1. **Instance/GPU detection**: reads the instance type and region via EC2 IMDSv2, and reads the
   GPU model, compute capability (sm_xx), and total VRAM via torch.
2. **Adaptive VRAM-saturation loop** (fully automatic):
   1. Starts by generating test audio at **20×** the base audio (`--audio-multiplier`).
   2. Sweeps batch_size and **keeps doubling upward to find the ceiling**: hitting an **OOM**
      (VRAM blown) or VRAM reaching the **90% target threshold** (`--vram-target-frac`) counts
      as saturated.
   3. If VRAM **stops rising as the batch grows yet is still not saturated** (diagnosed as
      **insufficient speech segments / segment starvation**), it automatically **multiplies the
      audio multiplier by 2** (`--audio-grow-factor`) and re-sweeps, until VRAM is saturated or
      a cap is reached (`--max-audio-multiplier` / `--max-iterations`).
3. **Metrics**: in the saturated state, computes RTF, median inference time, and peak VRAM for
   every successful batch size.
   - **Cost-optimal batch size ⭐**: highest RTF (lowest $/audio-hour).
   - **VRAM-saturated batch size 🔥**: highest peak VRAM (GPU most fully loaded).
4. **Pricing (three-level fallback)**: manual `--price` value → AWS Pricing API (requires
   boto3 + credentials) → built-in `prices.json`.
5. **Cost and reports**: computes $/audio-hour and the cost per 1,000 audio hours, and outputs
   JSON + Markdown.

> For the exact RTF calculation and the adaptive VRAM-saturation / segment-starvation algorithm
> (OOM boundary, VRAM plateau detection, automatic audio growth, etc.), see
> [docs/RTF-LOGIC.md](docs/RTF-LOGIC.md) (bilingual: English, with the Chinese original in
> collapsible blocks).

---

## Why Fixed float16 + Saturating VRAM on Each Machine

- **Fixed float16**: controls variables and keeps cross-instance comparison fair; moreover, on
  Blackwell (sm_120, e.g. G7/G7e) CTranslate2 **disables INT8**, so float16 is the common
  comparable baseline across GPU generations.
- **Each machine saturates its own VRAM**: the stronger the GPU and the larger its VRAM, the
  later its saturation point. If the audio / batch isn't enough to feed a strong GPU, that GPU
  will be systematically underestimated. Adaptive saturation ensures every machine is measured
  in a "fully loaded" state, which is what makes cross-machine comparison fair.

> ⚠️ Blackwell note: CTranslate2's tuning for sm_120 is still immature, and vendor-advertised
> compute does not necessarily translate fully into RTF — always rely on measured results. The
> project depends on the latest faster-whisper / CTranslate2 for Blackwell support.

> ⏱️ Runtime note: the longer the audio and the more iterations, the longer each sweep takes.
> Starting at 20× (about 5500s of audio) is usually enough to feed up to bs≈256. To run faster,
> lower `--audio-multiplier` or `--runs`; if segment starvation occurs, the audio is grown
> automatically.

---

## Common Options

```bash
python benchmark.py \
  --audio-multiplier 20 \      # starting audio = N× the base (default 20)
  --vram-target-frac 0.90 \    # VRAM utilization at which it counts as "saturated" (default 0.90)
  --audio-grow-factor 2.0 \    # growth factor of the audio multiplier on segment starvation (default 2.0)
  --max-audio-multiplier 1024 \# safety cap on the audio multiplier
  --max-batch 1024 \           # safety cap for upward batch probing
  --max-iterations 6 \         # max rounds of adaptive audio growth
  --batch-sizes 1,4,8,16,32,48,64,80,96 \  # starting candidate batches (saturate mode keeps doubling beyond them)
  --runs 1 \                   # timed runs per batch size, median taken (default 1)
  --price 2.52 \               # manually set the On-Demand hourly price (highest priority)
  --no-saturate \              # legacy mode: stop early on RTF saturation, don't chase full VRAM
  --audio audio/my.mp3         # specify the audio directly (skips auto-generation and adaptive growth)
```

Generate / pre-generate the test audio separately (optional; `benchmark.py` does this
automatically):

```bash
python prepare_audio.py --multiplier 32     # concatenate to 32× the base
python prepare_audio.py --target 900        # or concatenate to a target duration (seconds)
```

## Directory Layout

```
benchmark.py           Main entry: detect → adaptively saturate VRAM → compute RTF/time → compute cost → write reports
aggregate_report.py    Aggregates multi-instance results into a comparison report (with a VRAM utilization column)
prepare_audio.py       Generates/concatenates the standard test audio (build_audio is called by benchmark)
prices.json            On-Demand price fallback table (updatable)
setup.sh               One-step environment setup
requirements.txt       Dependencies (torch is installed separately from the matching CUDA wheel)
src/                   gpu_info / pricing / bench_core / report
audio/base-sample.mp3  Bundled base audio sample
results/               Per-instance reports and aggregated output
```

## Pricing Data

Prices in `prices.json` carry a `last_verified` date and change over time and by region. If AWS
credentials are configured in the runtime environment, **live Pricing API prices** are used
first; otherwise this table is used as the fallback. You can also override the price at any
time with `--price`. Treat the
[AWS EC2 On-Demand pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) as
authoritative.

## License

MIT, see `LICENSE`. The bundled `audio/base-sample.mp3` is synthetic call audio for demo
purposes; you can replace it with your own test audio (when comparing across instances, make
sure every machine uses the same file).
