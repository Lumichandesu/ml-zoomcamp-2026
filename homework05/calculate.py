#!/usr/bin/env python3
"""Verify and calculate ML Zoomcamp 2026 HW05 answers from the official release."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SOURCE_COMMIT = "5356b3b16e5c87f8f9fe0c3f8fdb2183d34b18fb"
MODEL_SHA256 = "1646bbdcd38d4f044da6b630c5b332c93a314245a8c21929011c42de51f629f1"
DEFAULTS_SHA256 = "1694f842b908fac9f53c0cc19be6d91f934befc63af23cc32fefafde69acd0cd"
Q3_LEAD = {
    "lead_source": "paid_ads",
    "industry": "technology",
    "employment_status": "employed",
    "location": "north_america",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0,
    "interaction_count": 4,
    "lead_score": 0.41,
}
Q4_LEAD = {
    "lead_source": "organic_search",
    "industry": "technology",
    "employment_status": "employed",
    "location": "europe",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0,
    "interaction_count": 7,
    "lead_score": 0.74,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_json(url: str, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"API request failed: {method} {url}: {exc}") from exc


def api_evidence(base_url: str) -> dict[str, Any]:
    base = base_url.rstrip("/")
    health = request_json(f"{base}/health")
    prediction = request_json(f"{base}/predict", "POST", Q4_LEAD)
    if health.get("status") != "ok":
        raise RuntimeError(f"unexpected /health response from {base}: {health}")
    if health.get("model_sha256") != MODEL_SHA256:
        raise RuntimeError(f"API model checksum does not match the verified release: {health}")
    if health.get("feature_defaults_sha256") != DEFAULTS_SHA256:
        raise RuntimeError(f"API feature-default checksum does not match the release: {health}")
    probability = prediction.get("conversion_probability")
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise RuntimeError(f"invalid conversion_probability from {base}: {prediction}")
    if not isinstance(prediction.get("conversion"), bool):
        raise RuntimeError(f"invalid conversion field from {base}: {prediction}")
    return {"base_url": base, "health": health, "prediction": prediction}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "official_dir",
        type=Path,
        help="Path to cohorts/2026/homework/05-deployment in a separate official checkout",
    )
    parser.add_argument("--api-url", help="Running local API base URL for Q4, such as http://localhost:9696")
    parser.add_argument("--docker-api-url", help="Running container API base URL for Q6, such as http://localhost:9697")
    parser.add_argument("--output", type=Path, help="Write the JSON evidence to this path")
    args = parser.parse_args()

    root = args.official_dir.resolve()
    model_path = root / "pipeline.bin"
    defaults_path = root / "feature_defaults.json"
    metadata_path = root / "model_metadata.json"
    lock_path = root / "uv.lock"
    dockerfile_path = root / "Dockerfile"
    for required in (model_path, defaults_path, metadata_path, lock_path, dockerfile_path, root / "model.py"):
        if not required.is_file():
            raise RuntimeError(f"missing official release file: {required}")

    source_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != SOURCE_COMMIT:
        raise RuntimeError(f"official checkout must be at {SOURCE_COMMIT}; found {source_commit}")

    model_hash = sha256(model_path)
    defaults_hash = sha256(defaults_path)
    if model_hash != MODEL_SHA256:
        raise RuntimeError(f"refusing to unpickle unexpected model: SHA256 {model_hash}")
    if defaults_hash != DEFAULTS_SHA256:
        raise RuntimeError(f"feature_defaults.json checksum mismatch: {defaults_hash}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    lock_text = lock_path.read_text(encoding="utf-8")
    docker_text = dockerfile_path.read_text(encoding="utf-8")
    locked_sklearn = re.search(r'(?ms)^name = "scikit-learn"\nversion = "([^"]+)"', lock_text)
    image_tag = re.search(r"(?m)^FROM python:([^@\s]+)", docker_text)
    if not locked_sklearn or not image_tag:
        raise RuntimeError("could not read sklearn version from uv.lock or Python image tag from Dockerfile")
    if locked_sklearn.group(1) != "1.7.2" or metadata.get("scikit_learn") != locked_sklearn.group(1):
        raise RuntimeError(f"reference sklearn versions disagree: lock={locked_sklearn.group(1)}, metadata={metadata.get('scikit_learn')}")
    if image_tag.group(1) != "3.11.15-slim-bookworm":
        raise RuntimeError(f"unexpected Python Docker base image tag: {image_tag.group(1)}")
    if metadata.get("model_sha256") != MODEL_SHA256:
        raise RuntimeError("model_metadata.json does not record the verified model checksum")

    uv = subprocess.run(["uv", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    code = (
        "import hashlib,json,pickle,sys; from pathlib import Path; from model import normalize_record; "
        f"p=Path({str(model_path)!r}); expected={MODEL_SHA256!r}; "
        "actual=hashlib.sha256(p.read_bytes()).hexdigest(); "
        "assert actual==expected, (actual,expected); "
        "m=pickle.load(p.open('rb')); "
        f"lead={json.dumps(Q3_LEAD)!r}; "
        "print(json.dumps({'python':sys.version.split()[0], 'scikit_learn':__import__('sklearn').__version__, "
        "'q3_probability':float(m.predict_proba([normalize_record(json.loads(lead))])[0,1])}))"
    )
    inference = subprocess.run(
        ["uv", "run", "--project", str(root), "python", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    q3 = json.loads(inference.stdout.strip().splitlines()[-1])
    if q3["scikit_learn"] != locked_sklearn.group(1):
        raise RuntimeError(f"runtime sklearn differs from lockfile: {q3['scikit_learn']}")

    result: dict[str, Any] = {
        "source_commit": source_commit,
        "official_source_subdirectory": "cohorts/2026/homework/05-deployment",
        "uv": uv,
        "python": q3["python"],
        "scikit_learn": q3["scikit_learn"],
        "docker_python_base_tag": f"python:{image_tag.group(1)}",
        "pipeline_sha256": model_hash,
        "feature_defaults_sha256": defaults_hash,
        "dataset_sha256_from_metadata": metadata.get("dataset_sha256"),
        "q3_probability": q3["q3_probability"],
        "q3_answer_3dp": f"{q3['q3_probability']:.3f}",
    }
    if args.api_url:
        api = api_evidence(args.api_url)
        result["q4_api"] = api
        result["q4_answer_3dp"] = f"{api['prediction']['conversion_probability']:.3f}"
    if args.docker_api_url:
        docker = api_evidence(args.docker_api_url)
        result["q6_container"] = docker
        result["q6_answer_3dp"] = f"{docker['prediction']['conversion_probability']:.3f}"

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
