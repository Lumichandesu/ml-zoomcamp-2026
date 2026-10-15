# ML Zoomcamp 2026 — Homework 5

This folder contains an original answer and verification script for Homework 5. It uses the cohort's frozen official deployment release; it does not redistribute the course's model, dataset, lockfile, copied specification, or generated environment files.

## Answers

| Question | Answer |
| --- | --- |
| 1 | `uv 0.12.7` on the verification machine (setup check) |
| 2 | Scikit-Learn `1.7.2` |
| 3 | Conversion probability `0.533` (raw local result: `0.532994`) |
| 4 | Conversion probability `0.770` (`conversion: true`; raw local API result: `0.769799`) |
| 5 | `python:3.11.15-slim-bookworm` |
| 6 | Conversion probability `0.770` (`conversion: true`; raw local container result: `0.769799`) |

## Provenance

The frozen release comes from the official DataTalksClub repository at commit [`5356b3b16e5c87f8f9fe0c3f8fdb2183d34b18fb`](https://github.com/DataTalksClub/machine-learning-zoomcamp/commit/5356b3b16e5c87f8f9fe0c3f8fdb2183d34b18fb), in `cohorts/2026/homework/05-deployment`. Before loading the pickle, the verifier checks `pipeline.bin` SHA256 against the assignment's published digest:

`1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1`

It also checks the official feature-defaults checksum, the lockfile and metadata Scikit-Learn version, and the Python image tag declared in the official Dockerfile. The dataset checksum recorded by the release metadata is `2d9da196bdefd2a45aa0a17ca21c19f06a5d9d56af628be1f04bbb3531cc12e9`.

The frozen model was trained for the 2026 assignment from `course_lead_scoring_2026.csv`, with the documented eight observable lead features. No prior-cohort model is used. The official model artifact is downloaded only by following the setup below; it is not included here.

## Reproduce

Clone the official repository separately and check out the pinned release commit:

```sh
git -c core.autocrlf=false clone https://github.com/DataTalksClub/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp
git checkout 5356b3b16e5c87f8f9fe0c3f8fdb2183d34b18fb
cd cohorts/2026/homework/05-deployment
uv sync --locked
uv run python smoke_test.py
```

From the submission repository root, run the original verifier, passing the official homework directory:

```sh
python homework05/calculate.py /path/to/machine-learning-zoomcamp/cohorts/2026/homework/05-deployment --output homework05/results.json
```

The script checks the release files before unpickling and computes Question 3 in the locked project environment. To collect Question 4 from an already running local API, pass `--api-url http://localhost:9696`. For Question 6, build and run the official container on a free local port, then pass its URL with `--docker-api-url`:

```sh
docker build -t zoomcamp-model:2026-hw5 .
docker run --rm -p 9697:9696 zoomcamp-model:2026-hw5
```

In another terminal:

```sh
python homework05/calculate.py /path/to/machine-learning-zoomcamp/cohorts/2026/homework/05-deployment --api-url http://localhost:9696 --docker-api-url http://localhost:9697 --output homework05/results.json
```

The API checks compare both health checksums to the frozen release and validate the prediction probability and boolean conversion field. Stop the local server/container with Ctrl+C when finished. The generated `results.json` is optional evidence and is not needed to run the verifier.

## Verification record

The values above were reproduced locally with Python `3.11.15`, `uv 0.12.7`, and locked Scikit-Learn `1.7.2`. The official `smoke_test.py` passed; the host API and a locally built Docker container each reported health status `ok`, the expected model and defaults checksums, and the same Question 4/6 result. The local Docker engine was Docker `29.8.2` on Linux. No cloud resources or paid services were used.
