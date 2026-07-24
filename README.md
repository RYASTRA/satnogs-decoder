# satnogs-decoder

**What does the frame say?** — a Kaitai `.ksy` decoder workbench for SatNOGS telemetry.

Adding a new satellite to SatNOGS's telemetry decoding means someone hand-writes a
[Kaitai Struct](https://kaitai.io) `.ksy` by hand — and that is a real wall for a satellite team.
The framing and transport around a beacon (AX.25, CSP) are well-trodden; the bespoke part, the part
that actually stops people, is the **payload field layout**: which bytes are which numbers, how
wide, signed or not, scaled by what. This repo is a workbench around exactly that wall. It does not
reinvent parsing — Kaitai does that — it helps you **author, check, and bootstrap** the `.ksy` that
turns a satellite's raw frames into labeled telemetry.

## The SatNOGS fleet

Four small, honest tools around a SatNOGS observation — three single-purpose engines, plus one app
that composes them for a human reviewer:

| repo | the question it answers |
|---|---|
| [satnogs-signal](https://github.com/RYSATNOGS/satnogs-signal) | *is there a signal in this waterfall?* — signal-vs-noise triage |
| **satnogs-decoder** (this repo) | ***what does the frame say?*** — telemetry decoding |
| [satnogs-id](https://github.com/RYSATNOGS/satnogs-id) | *which catalog object is it?* — Doppler identification |
| [satnogs-dashboard](https://github.com/RYSATNOGS/satnogs-dashboard) | *review it all on one observation* — the workbench that runs the three engines |

The three engines are standalone and read-only against SatNOGS; the dashboard is the surface that
composes them. This repo is the **decode** stage — raw frames in, telemetry fields out.

## How it works

The workbench gives you three ways at the payload-layout wall:

1. **generate** — write a compact *field-table spec* (YAML: offsets, widths, signedness, scaling,
   enums) and get complete, compilable SatNOGS `.ksy` text out — transport headers and
   switch/discriminator dispatch included.
2. **validate** — compile any `.ksy` and run it against a satellite's **live SatNOGS frames**,
   scoring `decode_rate` (did every frame parse?), byte-coverage (`mean_consumed_frac` /
   `full_coverage_rate`), and a name-based **cross-check** against the canonical
   [`satnogs-decoders`](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders) decode of
   the same frames.
3. **infer** — when there is no spec yet, a small trained model reads the raw payload bytes and
   proposes a **first-draft structural `.ksy`** (field boundaries + widths + signedness). Feed that
   straight into *validate* to see how far it got.

Everything is **read-only** against SatNOGS — the workbench fetches frames and reference decodes; it
never writes back.

## Does it work? (honest eval)

Three verbs at three very different maturities — reported straight, each against a live SatNOGS
anchor rather than a synthetic fixture.

**generate — clean GO.** The full pipeline (`generate()` → `compile_ksy()` → `get_fields()`) was
scored for **equivalence against the live upstream decoder** on VZLUSAT-2's beacon frame type
(NORAD 51085, `cmd == 0x56`):

| metric | value |
|---|---|
| frames fetched (live SatNOGS DB, 6 h window) | 54 |
| `decode_rate` (our v1-covered case) | **1.0000** (54 / 54) |
| cross-check `n_compared` | **540** (54 frames × 10 shared field names) |
| cross-check `agreement` | **1.0000** — no mismatched fields |

A genuine field-level equivalence result (not the vacuous `n_compared=0` fallback), passing at the
strongest threshold offered (`agreement ≥ 0.99`), with switch/discriminator dispatch and a scaling
instance both exercised.

**validate — GO.** On GRBAlpha (NORAD 47959), every live frame parsed and byte-coverage was
essentially complete; the validator also flags the two distinct corruption signals:

| metric | value |
|---|---|
| `decode_rate` | **1.0000** (7 / 7 live frames) |
| `mean_consumed_frac` | **0.997** |
| truncated frame | `decode_rate` drops — **flagged** |
| padded frame | `full_coverage_rate` drops — **flagged** |

**infer — early baseline, stated plainly.** The structural inference model, scored on a **held-out**
set of satellites, is exactly the rough-first-pass you'd expect and no better:

| metric | value |
|---|---|
| boundary recall / precision / F1 | 0.920 / 0.456 / **0.587** |
| exact-span F1 | **0.323** |
| signedness accuracy | 0.834 |
| enum precision / recall / F1 | **0.000** |
| over-segmentation rate | 0.451 |

It finds **most field starts** (92 % recall) but **over-segments heavily**, exact spans are weak,
and it does not recover enums or semantics at all. More frames don't help — the degradation curve is
flat at ~0.58 F1 from 30 frames to all — so this is feature-limited, not data-volume-limited. Treat
an inferred `.ksy` as a **structural skeleton to hand-correct**, not a finished decoder.

**Limitations, stated plainly:** `generate` and `validate` are each proven on a **single anchor /
one frame type** so far (VZLUSAT-2 beacon; GRBAlpha), and the v1 field-table class deliberately
excludes what the harder anchors need — magic-byte matches, `valid:` constraints, `repeat`, nested
switches, `process:`, and switches on *computed* bitfields (which is why `ledsat` and `duchifat3`
are out of v1 scope). `infer` is a research spike: structural-only, no semantics, and rough at that.

## Artifacts

Unlike the sibling [satnogs-signal](https://github.com/RYSATNOGS/satnogs-signal) (a published
model) and [satnogs-id](https://github.com/RYSATNOGS/satnogs-id) (a published dataset), this repo's
inference model is small enough to **live in the repo**: `satnogs_decoder/infer/model/*.joblib` — a
scikit-learn boundary / signed / enum trio plus metadata, loaded directly by `infer.py`. The
training **corpus** is deliberately *not* committed — it is temporary scaffolding (gitignored,
rebuilt on demand, deleted at finalization). When useful, a frame corpus can still be exported and
published as a Hugging Face Dataset:

```bash
docker compose run --rm app python scripts/build_corpus.py --limit 200 --push owner/satnogs-frames
```

## Running it (Docker — no virtualenv)

Everything runs in a container (`python:3.14-slim` with a JVM + the Kaitai Struct compiler baked
in); there is no host Python environment to manage. Put your SatNOGS DB token in a gitignored `.env`
(`satnogs_db_api_key=…` — frames are authenticated; `satnogs_network_api_key` and
`HUGGING_FACE_HUB_TOKEN` are optional). Compose loads it automatically.

```bash
docker compose build
make test        # or: docker compose run --rm app pytest

# generate a .ksy from a field-table spec (optionally validate it against live frames)
docker compose run --rm app python scripts/generate.py --spec my_sat.yaml --out my_sat.ksy
docker compose run --rm app python scripts/generate.py --spec my_sat.yaml --validate --anchor grbalpha

# validate an existing .ksy against a satellite's live SatNOGS frames
docker compose run --rm app python scripts/validate.py --ksy my_sat.ksy --anchor grbalpha

# infer a first-draft structural .ksy from raw frames (no spec needed), then validate it
docker compose run --rm app python scripts/infer.py --norad 47959 \
    --start 2022-06-15T00:00:00Z --end 2022-06-15T06:00:00Z --validate
```

`make build` / `test` / `shell` / `lint` wrap the common container commands.

## Layout

```
satnogs_decoder/
  shared/     ksc compile wrapper · .ksy helpers · reference decodes · SatNOGS DB client
  generate/   field-table spec → complete .ksy         (schema · fields · headers · build)
  validate/   compile + run a .ksy on live frames → decode_rate / coverage / cross-check
  infer/      structural inference from raw bytes       (corpus · features · layout · model …)
              + the frozen model that ships             (infer/model/*.joblib)
  data/       export a frame corpus as a Hugging Face Dataset
scripts/      thin CLI drivers — generate.py · validate.py · infer.py · build_corpus.py
```

## Credit

Parsing is done by **[Kaitai Struct](https://kaitai.io)** (`ksc` + the Python runtime). The
canonical `.ksy` decoders — and the reference decodes this workbench validates against — are **Libre
Space Foundation's [satnogs-decoders](https://gitlab.com/librespacefoundation/satnogs/satnogs-decoders)**.
Frames come from the **[SatNOGS](https://satnogs.org)** network and DB. This project is the
generate / validate / infer workbench around them — read-only; nothing is written back to SatNOGS.
MIT-licensed.
