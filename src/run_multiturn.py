#!/usr/bin/env python3
"""Detector-redaction (C7/C8) and sequential multi-turn runner (NEW-INFERENCE command; needs GPU + vLLM + the detector stack).

Moved from the original C7/C8 / adaptive-attack runner, plus the holdout case executor (execute_case, bottom of file).
Runs the two post-review experiments on the existing pipeline:

  C experiment (detector): conditions C7/C8 over the ORIGINAL 300 probes.
    C7 = redact only the spans a REAL detector (Presidio + spaCy) detected
         (from results_c/c7_redaction_map.jsonl) — the production version of
         oracle C2. Missed spans stay in the prompt.
    C8 = C7 + output filter.
  D experiment (adaptive attacks): conditions C0/C1 over the 150 adaptive
     probes (T7 encoded, T8 indirect, T9 multiturn_esc).
    T9 is generated as TRUE sequential multi-turn: turn1 -> model reply ->
    turn2 (+history) -> model reply -> turn3 (+history); leakage is scored
    over all generated turns.

Output layout: <results-dir>/raw/<Model>/<Cond>/<pid>.json
Record fields: model, condition, probe_id, probe_type, prompt (messages of
last turn), outputs_all (list of generated turns), output (last turn),
n_tokens, wall_ms, finish_reason, timestamp_utc.

Resumable: existing result file is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filter_output import apply_output_filter  # noqa: E402
from measure_utility import parse_json, restore_fields, prepare, score_detection, score_exposure, score_utility, interval_positions  # noqa: E402
from redact_input import REDACT_TOKENS, make_detector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

REFUSAL_SYSTEM_PROMPT = (ROOT / "prompts" / "refusal_system.txt").read_text().strip()

MODEL_CTX = {
    "Qwen/Qwen3-8B": 4096,
    "mistralai/Mistral-7B-Instruct-v0.3": 4096,
    "RedHatAI/Llama-3.1-8B-Instruct": 4096,
    "allenai/Olmo-3-7B-Instruct": 4096,
    "microsoft/phi-4": 8000,
    "google/gemma-4-E4B-it": 8192,
}
DEFAULT_MAX_MODEL_LEN = 4096
MODEL_EXTRA_ARGS = {
    "Qwen/Qwen3-8B": ["--default-chat-template-kwargs",
                      '{"enable_thinking": false}'],
}
SEQ_MULTI_TURN_TYPES = {"multiturn_esc"}  # true sequential generation


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def display_name(repo: str) -> str:
    return repo.split("/")[-1]


class VLLMServer:
    def __init__(self, repo: str, port: int, logdir: Path,
                 gpu_mem_util: float, max_model_len: int, hf_home: str,
                 workdir: str = "."):
        self.repo = repo
        self.port = port
        self.gpu_mem_util = gpu_mem_util
        self.workdir = workdir
        self.max_model_len = min(max_model_len, MODEL_CTX.get(repo,
                                                              DEFAULT_MAX_MODEL_LEN))
        self.hf_home = hf_home
        self.logpath = logdir / f"cd_server_{display_name(repo)}.log"
        self.proc = None

    def start(self):
        assert self.proc is None
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
        ] + MODEL_EXTRA_ARGS.get(self.repo, [])
        env = dict(os.environ)
        env.update({
            "HF_HOME": self.hf_home,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
            "PYTHONUNBUFFERED": "1",
            "VLLM_USE_FLASHINFER_SAMPLING": "0",
            "PATH": str(Path(sys.executable).parent)
                    + os.pathsep + env.get("PATH", ""),
        })
        log(f"starting vLLM: {self.repo}")
        with open(self.logpath, "a") as lf:
            self.proc = subprocess.Popen(
                cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                cwd=self.workdir, start_new_session=True,
            )
        self.wait_ready(timeout=900)

    def _kill(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=60)
        except Exception:
            try:
                self.proc.kill()
                self.proc.wait(timeout=30)
            except Exception:
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
                    f"server exited rc={self.proc.returncode}\n{tail}")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return
            except Exception as e:
                last_err = str(e)
            time.sleep(3)
        raise RuntimeError(f"server not ready in {timeout}s; last: {last_err}")

    def healthy(self) -> bool:
        if self.proc is None or self.proc.poll() is not None:
            return False
        try:
            import urllib.request
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def stop(self):
        log("stopping vLLM server")
        self._kill()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def redact_turns(turns: list[dict], det_spans: list[dict]) -> list[str]:
    """Replace detector-reported spans (joined-prompt coordinates) with the
    type token — the realistic production behavior: only what the detector
    saw is redacted. Spans are applied longest-first, non-overlapping,
    right-to-left within each turn so offsets stay valid.

    Returns the list of (possibly redacted) turn contents.
    """
    contents = [t["content"] for t in turns]
    offsets = []
    pos = 0
    for c in contents:
        offsets.append(pos)
        pos += len(c) + 1  # "\n" joiner

    # attribute each span to the turn containing its start
    by_turn: dict[int, list[dict]] = {}
    for s in det_spans:
        for i, (o, cl) in enumerate(zip(offsets, contents)):
            if o <= s["start"] < o + len(cl):
                by_turn.setdefault(i, []).append(s)
                break
    red = list(contents)
    for ti, ss in by_turn.items():
        # right-to-left so earlier replacements don't shift later offsets
        for s in sorted(ss, key=lambda x: -x["start"]):
            o = offsets[ti]
            a, b = s["start"] - o, s["end"] - o
            if not (0 <= a < b <= len(red[ti])):
                continue  # span crosses the turn boundary; skip (rare)
            red[ti] = red[ti][:a] + REDACT_TOKENS[s["type"]] + red[ti][b:]
    return red


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--redact-map", default="",
                    help="optional precomputed detector map (rows {probe_id, spans}); if omitted, C7/C8 run the detector live on the joined prompt text")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--gpu-mem-util", type=float, default=0.55)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--jobs", action="append", required=True)
    ap.add_argument("--conditions", required=True)
    args = ap.parse_args()

    results = Path(args.results_dir)
    raw_dir = results / "raw"
    log_dir = results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for spec in args.jobs:
        repo, _, quant = spec.partition("::")
        jobs.append((repo, quant or "BF16"))
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    probes = load_jsonl(Path(args.probes))

    c7_map = {}
    if args.redact_map:
        for row in load_jsonl(Path(args.redact_map)):
            c7_map[row["probe_id"]] = row.get("spans", [])

    if not c7_map and any(c in ("C7", "C8") for c in conds):
        detect = make_detector()  # live detection on the joined prompt text (the detector-evaluation protocol)
        for p in probes:
            c7_map[p["probe_id"]] = detect("\n".join(t["content"] for t in p["turns"]))

    total = len(jobs) * len(conds) * len(probes)
    log(f"PLAN: {len(jobs)} models x {len(conds)} conds x {len(probes)} probes "
        f"= {total} cell inferences (T9 cells generate 3 completions)")

    import openai

    done = 0
    t_start = time.time()
    for job_idx, (repo, quant) in enumerate(jobs, 1):
        mname = display_name(repo) if quant == "BF16" else f"{display_name(repo)}-{quant}"
        mdir = raw_dir / mname
        for c in conds:
            (mdir / c).mkdir(parents=True, exist_ok=True)
        log(f"=== job {job_idx}/{len(jobs)}: {repo} [{quant}] ===")
        server = VLLMServer(repo, args.port, log_dir, args.gpu_mem_util,
                            MODEL_CTX.get(repo, DEFAULT_MAX_MODEL_LEN),
                            args.hf_home, args.workdir)
        client = None
        try:
            server.start()
            log(f"server ready: {repo}")
            client = openai.OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1",
                                   api_key="EMPTY")

            def complete(messages) -> tuple[str, str, int]:
                last_err = None
                for attempt in range(6):
                    try:
                        resp = client.chat.completions.create(
                            model=repo, messages=messages,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens)
                        return (resp.choices[0].message.content or "",
                                resp.choices[0].finish_reason,
                                resp.usage.completion_tokens
                                if resp.usage else None)
                    except Exception as e:
                        last_err = e
                        log(f"request error (attempt {attempt+1}): {e}")
                        time.sleep(5 * (attempt + 1))
                raise RuntimeError(f"request failed: {last_err}")

            for probe in probes:
                pid = probe["probe_id"]
                detected = c7_map.get(pid, [])
                for c in conds:
                    out_path = mdir / c / f"{pid}.json"
                    if out_path.exists():
                        done += 1
                        continue
                    t0 = time.time()
                    do_refusal = c in ("C1",)
                    do_det_redact = c in ("C7", "C8")
                    do_filter = c in ("C8",)
                    base_sys = ([{"role": "system",
                                  "content": REFUSAL_SYSTEM_PROMPT}]
                                if do_refusal else [])
                    outputs_all = []
                    finish = ""
                    n_tok = 0
                    last_messages = []
                    if probe["type"] in SEQ_MULTI_TURN_TYPES:
                        # true sequential multi-turn with generated replies
                        red_contents = (redact_turns(probe["turns"], detected)
                                        if do_det_redact
                                        else [t["content"] for t in probe["turns"]])
                        history = list(base_sys)
                        for turn, content in zip(probe["turns"], red_contents):
                            history.append({"role": "user", "content": content})
                            out, finish, tok = complete(history)
                            if do_filter:
                                out, _ = apply_output_filter(out)
                            outputs_all.append(out)
                            n_tok = (n_tok or 0) + (tok or 0)
                            history.append({"role": "assistant",
                                            "content": out})
                    else:
                        red_contents = (redact_turns(probe["turns"], detected)
                                        if do_det_redact
                                        else [t["content"] for t in probe["turns"]])
                        msgs = list(base_sys)
                        for turn, content in zip(probe["turns"], red_contents):
                            msgs.append({"role": "user", "content": content})
                        out, finish, tok = complete(msgs)
                        if do_filter:
                            out, _ = apply_output_filter(out)
                        outputs_all = [out]
                        n_tok = tok
                        last_messages = msgs
                    wall_ms = int((time.time() - t0) * 1000)
                    rec = {
                        "model": mname,
                        "model_repo": repo,
                        "condition": c,
                        "quant": quant,
                        "probe_id": pid,
                        "probe_type": probe["type"],
                        "encoded": probe.get("encoded", {}),
                        "n_turns_generated": len(outputs_all),
                        "prompt": last_messages or list(base_sys),
                        "outputs_all": outputs_all,
                        "output": outputs_all[-1],
                        "n_tokens": n_tok,
                        "wall_ms": wall_ms,
                        "finish_reason": finish,
                        "timestamp_utc": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    tmp = out_path.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(rec))
                    tmp.replace(out_path)
                    done += 1
                    rate = done / (time.time() - t_start) * 60
                    log(f"[{done}/{total}] {mname}|{c}|{pid} "
                        f"turns={len(outputs_all)} {wall_ms}ms | {rate:.1f}/min")
                    if not server.healthy():
                        log("server unhealthy; restarting")
                        server.stop()
                        server.start()
                        client = openai.OpenAI(
                            base_url=f"http://127.0.0.1:{args.port}/v1",
                            api_key="EMPTY")
        except Exception as e:
            log(f"JOB FAILED ({repo} [{quant}]): {e}")
            (results / f"cd_job_failed_{mname}.json").write_text(
                json.dumps({"repo": repo, "quant": quant, "error": str(e)}))
        finally:
            server.stop()
            time.sleep(10)

    log(f"ALL DONE: {done}/{total} in { (time.time()-t_start)/60:.1f} min")



# ---- holdout case executor (formerly holdout_utility/runner.py): sequential multi-turn with C7 history re-redaction
import copy  # noqa: E402
import urllib.parse  # noqa: E402
import urllib.request  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

# frozen holdout sampling / per-model chat-template kwargs (the original frozen holdout configuration)
HOLDOUT_SAMPLING = {'temperature': 0.2, 'top_p': 1.0, 'max_tokens': 256, 'n': 1, 'stream': False}
HOLDOUT_CHAT_TEMPLATE_KWARGS = {'Qwen/Qwen3-8B': {'enable_thinking': False}}


def validate_endpoint(endpoint):
    u = urllib.parse.urlsplit(endpoint)
    if u.scheme != 'http' or u.hostname not in ('127.0.0.1', '::1') or u.username or u.password or u.query or u.fragment:
        raise ValueError('Only numeric loopback HTTP endpoints without credentials are permitted')
    if u.path != '/v1/chat/completions':
        raise ValueError('Use explicit /v1/chat/completions endpoint')
    return endpoint


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Endpoint redirects prohibited')


class LocalCompletion:
    def __init__(self, endpoint, model, ledger=None):
        self.endpoint = validate_endpoint(endpoint)
        self.model = model
        self.ledger = ledger
        self.attempt_count = 0
        self.success_count = 0
        self.context = {}
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def __call__(self, messages, seed):
        payload = dict(HOLDOUT_SAMPLING, model=self.model, seed=seed, messages=messages,
                       chat_template_kwargs=HOLDOUT_CHAT_TEMPLATE_KWARGS.get(self.model, {}))
        req = urllib.request.Request(self.endpoint, json.dumps(payload).encode(),
                                     {'Content-Type': 'application/json'}, method='POST')
        self.attempt_count += 1
        event = dict(self.context, attempt=self.attempt_count, seed=seed, request=payload,
                     timestamp_utc=datetime.now(timezone.utc).isoformat())
        if self.ledger:
            self.ledger(dict(event, status='REQUESTED'))
        try:
            with self.opener.open(req, timeout=120) as response:
                result = parse_json(response.read().decode())
            if len(result.get('choices', [])) != 1:
                raise ValueError('Expected one completion choice')
            choice = result['choices'][0]
            output = choice['message']['content']
            if not isinstance(output, str):
                raise ValueError('Missing assistant answer content (not reasoning tokens)')
        except Exception as error:
            if self.ledger:
                self.ledger(dict(event, status='FAILED', error=str(error), error_type=type(error).__name__))
            raise
        self.success_count += 1
        response = {'output': output, 'finish_reason': choice.get('finish_reason'),
                    'usage': result.get('usage', {}), 'request': payload}
        if self.ledger:
            self.ledger(dict(event, status='RETURNED', response=response))
        return response


def _redact_history(history, defense):
    redacted = []
    audit = []
    for turn in history:
        result = defense.redact_text(turn['content'])
        if not isinstance(result, dict) or not isinstance(result.get('text'), str) or not isinstance(result.get('spans'), list):
            raise ValueError('redact_text adapter must return {text:str,spans:list}')
        interval_positions(turn['content'], result['spans'])
        redacted.append(dict(turn, content=result['text']))
        audit.append({'role': turn['role'], 'source': turn['content'], 'model_content': result['text'], 'detected_spans': result['spans']})
    return redacted, audit


def execute_case(case, condition, seed, completion, defense=None):
    if condition not in ('C0', 'C2_unique', 'C7'):
        raise ValueError('C8 is deterministic postprocessing, not a generation condition')
    turns, mapping = prepare(case) if condition == 'C2_unique' else (copy.deepcopy(case['turns']), {})
    history, requests, outputs, filtered, reasons, usage, audits = [], [], [], [], [], [], []
    for turn in turns:
        history.append(turn)
        messages = copy.deepcopy(history)
        if condition == 'C7':
            if defense is None:
                raise ValueError('Corrected detector required')
            messages, audit = _redact_history(messages, defense)
            audits.append(audit)
        if condition == 'C2_unique':
            # Gold known values in prior assistant replies are also refreshed.
            for message in messages:
                if message['role'] == 'assistant':
                    for token, e in sorted(mapping.items(), key=lambda kv: -len(kv[1]['text'])):
                        message['content'] = message['content'].replace(e['text'], token)
        response = completion(messages, seed)
        output = response['output']
        if not isinstance(output, str):
            raise TypeError('Completion must return assistant output text')
        outputs.append(output)
        if condition == 'C7':
            if defense is None:
                raise ValueError('Corrected filter required')
            observable = defense.filter_text(output)
            if not isinstance(observable, str):
                raise TypeError('filter_text adapter must return string')
            filtered.append(observable)
        requests.append(messages)
        reasons.append(response.get('finish_reason'))
        usage.append(response.get('usage', {}))
        history.append({'role': 'assistant', 'content': output})
    views = {condition: outputs}
    if condition == 'C7':
        views['C8'] = filtered
    utility, restoration_status = {}, 'not_requested'
    if case.get('expected') is not None:
        for view, observed in views.items():
            utility[view] = {'pre_restoration': score_utility(case, observed[-1])}
            if condition == 'C2_unique' and case.get('authorized_fields'):
                try:
                    restored = restore_fields(parse_json(observed[-1]), mapping, case['authorized_fields'], authorized=True)
                    utility[view]['authorized_recipient'] = score_utility(case, json.dumps(restored, ensure_ascii=False))
                    # Restored PII is not written to an attacker-visible result channel.
                    restoration_status = 'authorized_fields_restored_in_trusted_memory'
                except (ValueError, TypeError):
                    utility[view]['authorized_recipient'] = score_utility(case, observed[-1])
                    restoration_status = 'restoration_failed'
    detector_metrics = []
    if condition == 'C7':
        for index, audit in enumerate(audits):
            last = audit[-1]  # newly supplied user turn
            gold = [a for a in case['annotations'] if a['turn'] == index]
            detector_metrics.append(score_detection(last['source'], gold, last['detected_spans']))
    return {'case_id': case['case_id'], 'condition': condition, 'seed': seed,
            'generation_calls': len(outputs), 'model_requests': requests,
            'raw_assistant_outputs': outputs, 'observable_outputs': views,
            'exposure': {v: score_exposure(case['entities'], o) for v, o in views.items()},
            'utility': utility, 'restoration_status': restoration_status,
            'authorized_mapping_persisted': False, 'redaction_audit': audits,
            'detector_interval_metrics': detector_metrics, 'finish_reasons': reasons, 'usage': usage,
            'history_policy': 'raw assistant replay, refreshed input redaction each request; C8 derived without filtered-feedback generation'}


if __name__ == "__main__":
    main()
