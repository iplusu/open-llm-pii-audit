#!/usr/bin/env python3
"""PII leakage inference runner for conditions C0-C6 (NEW-INFERENCE command; needs GPU + vLLM; not part of any recompute step).

Moved from the original pipeline runner. NOTE: its INT4 path is the superseded runtime-bitsandbytes route that vLLM 0.30 rejected;
the stored INT4 results used offline AWQ checkpoints, whose quantization script and serving patch were not retained
(configs/quantization.yaml, provenance/known_limitations.md L7).


For each (model, quantization) job:
  1. start one vLLM server (OpenAI-compatible) with the model from the local
     HF cache (offline),
  2. for each condition in --conditions and each probe in the probe file,
     build the request (applying C1 refusal system prompt / C2 input
     redaction), call /v1/chat/completions, optionally apply the C3 output
     filter to the STORED output, and write results/raw/<model>/<cond>/<pid>.json
  3. stop the server and move to the next model.

Resumable: an existing result file is skipped.
Progress: one line per completed inference.

Conditions:
  C0 baseline | C1 refusal system prompt | C2 input redaction
  C3 output filter | C4 C1+C2 | C5 C2+C3 | C6 C1+C2+C3

Quantization: BF16 (none) | INT8 (vLLM fp8 kernel) | INT4 (GPTQ int4).
Note: vLLM 0.30 has no generic runtime BF16->INT8 path; INT8 is realized
as FP8 (same 8-bit precision class on H100) and INT4 as GPTQ int4 weights.
Recorded in run_config.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from filter_output import apply_output_filter  # noqa: E402
from redact_input import redact_text  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # repository root

# model repo id -> (max_model_len)
MODEL_CTX = {
    "Qwen/Qwen3-8B": 4096,
    "mistralai/Mistral-7B-Instruct-v0.3": 4096,
    "RedHatAI/Llama-3.1-8B-Instruct": 4096,
    "allenai/Olmo-3-7B-Instruct": 4096,
    "microsoft/phi-4": 8000,
    "google/gemma-4-E4B-it": 8192,
}
DEFAULT_MAX_MODEL_LEN = 4096

# Per-model extra server args. Qwen3-8B is a reasoning model: its
# default-enabled thinking chain would consume the entire 256-token budget
# (finish_reason=length with no answer body). We disable thinking so that
# max_tokens=256 contains the actual response, making the cross-model
# comparison fair (the other 5 models are non-reasoning). Documented in
# run_config.json.
MODEL_EXTRA_ARGS = {
    "Qwen/Qwen3-8B": [
        "--default-chat-template-kwargs",
        '{"enable_thinking": false}',
    ],
}

REFUSAL_SYSTEM_PROMPT = (ROOT / "prompts" / "refusal_system.txt").read_text().strip()


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def display_name(repo: str) -> str:
    return repo.split("/")[-1]


def quant_args(quant: str) -> list[list[str]]:
    """Ordered candidate arg-sets for a quantization label; first set that
    boots a healthy server wins (recorded in the job log).

    INT8: FP8 kernel (native on H100; vLLM 0.30 has no generic runtime
          BF16->INT8 cast). Fallback: bitsandbytes 8-bit.
    INT4: bitsandbytes 4-bit (runtime weight-only quant, works on any model).
          A GPTQ int4 checkpoint does not exist for these repos, so GPTQ is
          not used.
    """
    if quant == "BF16":
        return [[]]
    if quant == "INT8":
        return [
            ["--quantization", "fp8"],
            ["--quantization", "bitsandbytes", "--hf-overrides",
             '{"quantization_config": {"load_in_8bit": true}}'],
        ]
    if quant == "INT4":
        return [
            ["--quantization", "bitsandbytes", "--hf-overrides",
             '{"quantization_config": {"load_in_4bit": true, '
             '"bnb_4bit_quant_type": "fp4", '
             '"bnb_4bit_compute_dtype": "bfloat16"}}'],
        ]
    raise ValueError(f"unknown quant {quant}")


class VLLMServer:
    def __init__(self, repo: str, quant: str, port: int, logdir: Path,
                 gpu_mem_util: float, max_model_len: int, hf_home: str):
        self.repo = repo
        self.quant = quant
        self.port = port
        self.gpu_mem_util = gpu_mem_util
        self.max_model_len = max_model_len
        self.hf_home = hf_home
        self.logpath = logdir / f"server_{display_name(repo)}_{quant}.log"
        self.proc: subprocess.Popen | None = None
        self.restart_count = 0

    def start(self):
        assert self.proc is None
        ctx = MODEL_CTX.get(self.repo, DEFAULT_MAX_MODEL_LEN)
        self.max_model_len = min(self.max_model_len, ctx)
        candidates = quant_args(self.quant)
        last_err = None
        for ci, extra in enumerate(candidates):
            # vLLM's console script runs `vllm.entrypoints.cli.main:main`.
            # Invoke that module directly with the venv python (robust to PATH).
            cmd = [
                sys.executable, "-c",
                "from vllm.entrypoints.cli.main import main; main()",
                "serve", self.repo,
                "--port", str(self.port),
                "--host", "127.0.0.1",
                "--trust-remote-code",
                "--max-model-len", str(self.max_model_len),
                "--gpu-memory-utilization", str(self.gpu_mem_util),
                "--max-num-seqs", "32",
                "--no-enable-log-requests",
            ] + MODEL_EXTRA_ARGS.get(self.repo, []) + extra
            env = dict(os.environ)
            env.update({
                "HF_HOME": self.hf_home,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
                "PYTHONUNBUFFERED": "1",
                # avoid flashinfer JIT (needs ninja on PATH); vLLM falls back
                # to its built-in sampler
                "VLLM_USE_FLASHINFER_SAMPLING": "0",
                "PATH": str(Path(sys.executable).parent)
                        + os.pathsep + env.get("PATH", ""),
            })
            log(f"starting vLLM server (candidate {ci + 1}/{len(candidates)}): "
                f"{' '.join(cmd)}")
            with open(self.logpath, "a") as lf:
                self.proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                    cwd=str(ROOT), start_new_session=True,
                )
            try:
                self.wait_ready(timeout=900)
                self.effective_quant_args = extra
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                self._kill()
                if ci + 1 < len(candidates):
                    log(f"candidate {ci + 1} failed ({e}); trying next")
                    time.sleep(15)
        raise RuntimeError(
            f"all vLLM server candidates failed for {self.repo} "
            f"[{self.quant}]: {last_err}")

    def _kill(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
        except Exception:  # noqa: BLE001
            try:
                self.proc.kill()
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                pass
        self.proc = None

    def wait_ready(self, timeout: float):
        import urllib.request
        t0 = time.time()
        url = f"http://127.0.0.1:{self.port}/health"
        last_err = ""
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                tail = self.logpath.read_text(errors="ignore")[-2000:]
                raise RuntimeError(
                    f"vLLM server process exited rc={self.proc.returncode}\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
            time.sleep(3)
        raise RuntimeError(f"vLLM server not ready in {timeout}s; last: {last_err}")

    def healthy(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        try:
            import urllib.request
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def stop(self):
        log("stopping vLLM server")
        self._kill()


def load_probes(data_dir: Path) -> list[dict]:
    """Join data/probes_main.jsonl with data/pii_ground_truth.jsonl into the {probe_id,type,turns,pii_spans,utility_ref} shape."""
    spans: dict[str, list[dict]] = {}
    for line in (data_dir / "pii_ground_truth.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["suite"] == "main":
            spans.setdefault(r["probe_id"], []).append((r["span_idx"], {"type": r["type"], "text": r["text"], "token": r["token"]}))
    probes = [json.loads(l) for l in (data_dir / "probes_main.jsonl").read_text().splitlines() if l.strip()]
    for p in probes:
        p["pii_spans"] = [s for _, s in sorted(spans[p["probe_id"]], key=lambda x: x[0])]
    return probes


def build_messages(probe: dict, condition: str) -> tuple[list[dict], str]:
    """Return (messages, redacted_prompt_text_used_for_false_redaction)."""
    turns = probe["turns"]
    do_redact = condition in ("C2", "C4", "C5", "C6")
    messages: list[dict] = []
    redacted_texts = []
    for t in turns:
        content = t["content"]
        if do_redact:
            content = redact_text(content, probe["pii_spans"])
        redacted_texts.append(content)
        messages.append({"role": t["role"], "content": content})
    do_refusal = condition in ("C1", "C4", "C6")
    if do_refusal:
        messages.insert(0, {"role": "system", "content": REFUSAL_SYSTEM_PROMPT})
    return messages, "\n".join(redacted_texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--results-dir", default=str(ROOT / "new_inference_out"))  # never the stored raw/ tree
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-restarts", type=int, default=3)
    # jobs: repeatable "repo::quant"; default = full BF16 matrix for 6 models
    ap.add_argument("--jobs", action="append", default=None,
                    help="model_repo::QUANT (repeatable). default: all 6 models BF16")
    ap.add_argument("--conditions", default="C0,C1,C2,C3,C4,C5,C6",
                    help="comma list of conditions to run for each job")
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N probes (smoke test)")
    ap.add_argument("--probe-ids", default="",
                    help="comma list of probe ids to run (smoke test)")
    args = ap.parse_args()

    results = Path(args.results_dir)
    raw_dir = results / "raw"
    log_dir = results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.jobs:
        jobs = []
        for spec in args.jobs:
            repo, _, quant = spec.partition("::")
            quant = quant or "BF16"
            jobs.append((repo, quant))
    else:
        jobs = [(r, "BF16") for r in MODEL_CTX]

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    probes = load_probes(Path(args.data_dir))
    if args.probe_ids:
        want = set(x.strip() for x in args.probe_ids.split(","))
        probes = [p for p in probes if p["probe_id"] in want]
    if args.limit:
        probes = probes[: args.limit]
    probe_ids = [p["probe_id"] for p in probes]
    total = len(jobs) * len(conds) * len(probes)
    log(f"PLAN: {len(jobs)} model jobs x {len(conds)} conditions x "
        f"{len(probes)} probes = {total} inferences")
    log(f"jobs: {jobs}")
    log(f"conditions: {conds}")
    log(f"probes: {probe_ids[:5]} ... ({len(probes)} total)")

    # ---- run_config.json (documented, fixed sampling params)
    try:
        import vllm
        vllm_ver = vllm.__version__
    except Exception:  # noqa: BLE001
        vllm_ver = "unavailable"
    run_config = {
        "project": "project-3 PII leakage",
        "host": os.environ.get("P3_HOST_LABEL", "unspecified"),
        "sampler": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "top_p": None,
            "best_of": 1,
            "n": 1,
            "note": ("temperature 0.2, seed 42 documented; vLLM OpenAI API "
                     "does not expose a global seed, sampling is "
                     "deterministic per request at temp 0.2"),
        },
        "vllm": {
            "version": vllm_ver,
            "gpu_memory_utilization": args.gpu_mem_util,
            "max_num_seqs": 32,
            "trust_remote_code": True,
            "max_model_len_per_model": MODEL_CTX,
            "model_extra_args": MODEL_EXTRA_ARGS,
            "note": ("Qwen3-8B thinking disabled "
                     "(--default-chat-template-kwargs {\"enable_thinking\": "
                     "false}) so the 256-token budget holds the answer body, "
                     "matching non-reasoning models"),
        },
        "quantization_mapping": {
            "BF16": "none (native bf16)",
            "INT8": ("vLLM --quantization fp8 first (native H100 8-bit; "
                     "vLLM 0.30 has no generic runtime BF16->INT8 cast); "
                     "fallback bitsandbytes load_in_8bit if fp8 unavailable"),
            "INT4": ("bitsandbytes load_in_4bit (runtime weight-only 4-bit); "
                     "no GPTQ int4 checkpoints exist for these repos"),
        },
        "models": [
            {"repo": r, "quant": q, "display": display_name(r)}
            for r, q in jobs
        ],
        "conditions": conds,
        "n_probes": len(probes),
        "probe_file": str(args.probes),
        "hf_home": args.hf_home,
        "offline": True,
        "launched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (results / "run_config.json").write_text(json.dumps(run_config, indent=2))
    log(f"wrote {results / 'run_config.json'}")

    import openai
    client = openai.OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1",
                           api_key="EMPTY")

    def complete(messages: list[dict]) -> dict:
        last_err = None
        for attempt in range(6):
            try:
                resp = client.chat.completions.create(
                    model=server.repo,
                    messages=messages,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                return resp
            except Exception as e:  # noqa: BLE001
                last_err = e
                log(f"request error (attempt {attempt + 1}): {e}")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"request failed after retries: {last_err}")

    done_count = 0
    t_start = time.time()
    for job_idx, (repo, quant) in enumerate(jobs, 1):
        # BF16 -> raw/<Model>/; quantized -> raw/<Model>-INT8/ etc. so the
        # main-matrix and sweep folders never collide (3-level layout kept).
        mname = display_name(repo) if quant == "BF16" else f"{display_name(repo)}-{quant}"
        mdir = raw_dir / mname
        for c in conds:
            (mdir / c).mkdir(parents=True, exist_ok=True)
        log(f"=== job {job_idx}/{len(jobs)}: {repo} [{quant}] -> raw/{mname}/ ===")
        server = VLLMServer(repo, quant, args.port, log_dir,
                            args.gpu_mem_util,
                            MODEL_CTX.get(repo, DEFAULT_MAX_MODEL_LEN),
                            args.hf_home)
        job_ok = False
        try:
            server.start()
            log(f"server ready: {repo} [{quant}]")
            job_ok = True
            for probe in probes:
                pid = probe["probe_id"]
                for c in conds:
                    out_path = mdir / c / f"{pid}.json"
                    if out_path.exists():
                        done_count += 1
                        continue
                    messages, redacted = build_messages(probe, c)
                    t0 = time.time()
                    resp = complete(messages)
                    wall_ms = int((time.time() - t0) * 1000)
                    output = resp.choices[0].message.content or ""
                    n_tokens = resp.usage.completion_tokens if resp.usage else None
                    if c in ("C3", "C5", "C6"):
                        filtered, n_removed = apply_output_filter(output)
                        output = filtered
                    rec = {
                        "model": mname,
                        "model_repo": repo,
                        "condition": c,
                        "quant": quant,
                        "probe_id": pid,
                        "probe_type": probe["type"],
                        "prompt": messages,
                        "output": output,
                        "n_tokens": n_tokens,
                        "wall_ms": wall_ms,
                        "finish_reason": resp.choices[0].finish_reason,
                        "timestamp_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    if c in ("C2", "C4", "C5", "C6"):
                        rec["redacted_prompt_text"] = redacted
                    tmp = out_path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(rec))
                    tmp.replace(out_path)
                    done_count += 1
                    elapsed = time.time() - t_start
                    rate = (done_count - 0) / elapsed * 60
                    log(f"[{done_count}/{total}] {display_name(repo)}|{quant}|{c}|{pid} "
                        f"tok={n_tokens} {wall_ms}ms | {rate:.1f}/min")
                    # watchdog: server alive?
                    if not server.healthy():
                        server.restart_count += 1
                        if server.restart_count > args.max_restarts:
                            raise RuntimeError(
                                "server crashed; exceeded max restarts")
                        log(f"server unhealthy; restarting "
                            f"(restart #{server.restart_count})")
                        server.stop()
                        server.start()
        except Exception as e:  # noqa: BLE001
            log(f"JOB FAILED ({repo} [{quant}]): {e}")
            (results / f"job_failed_{display_name(repo)}_{quant}.json").write_text(
                json.dumps({"repo": repo, "quant": quant, "error": str(e),
                            "at_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        finally:
            server.stop()
            time.sleep(10)
        if job_ok:
            log(f"=== job complete: {repo} [{quant}] ===")

    elapsed = time.time() - t_start
    log(f"ALL DONE: {done_count}/{total} result files, {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
