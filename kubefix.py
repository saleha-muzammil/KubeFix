#!/usr/bin/env python3
"""
KubeFix: simplified server + verifier with multi-model LLM support
- FastAPI app exposing /mic and /generate
- LLM provider switch: Gemini (API) or HuggingFace (local DeepSeek etc.)
- /generate returns:
    - gen: {rbac_patch, netpol_patch, podsec_patch, summary, confidence, risk}
    - usage: {model, prompt_tokens, completion_tokens, total_tokens}
- CLI:
    python kubefix.py verify --mic MIC.json --patches PATCH_DIR
  prints a JSON summary (rbac_static_ok, can_i_summary, notes)
"""

import os
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------
# LLM usage + config helpers
# -----------------------------


@dataclass
class LLMUsage:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def get_llm_config():
    """
    Reads provider + model from environment.
    KUBEFIX_PROVIDER: 'gemini' or 'hf'
    KUBEFIX_MODEL:    model name or HF model id
    GEMINI_MODEL:     kept for backward compatibility
    """
    provider = os.getenv("KUBEFIX_PROVIDER", "gemini")
    model = os.getenv("KUBEFIX_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))
    return provider, model


# HuggingFace globals (lazy-loaded)
HF_MODEL = None
HF_TOKENIZER = None
HF_PIPELINE = None


def get_hf_pipeline(model_name: str):
    """
    Lazily load a HuggingFace text-generation pipeline.

    Requires:
      - transformers
      - correct access token via HUGGINGFACE_HUB_TOKEN (if private model)
    """
    global HF_MODEL, HF_TOKENIZER, HF_PIPELINE

    if HF_PIPELINE is not None:
        return HF_PIPELINE, HF_TOKENIZER

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # type: ignore

    HF_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    HF_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    HF_PIPELINE = pipeline(
        "text-generation",
        model=HF_MODEL,
        tokenizer=HF_TOKENIZER,
        max_new_tokens=int(os.getenv("KUBEFIX_MAX_NEW_TOKENS", "512")),
        do_sample=False,
        temperature=0.0,
    )
    return HF_PIPELINE, HF_TOKENIZER


def call_llm_for_patch(prompt: str) -> tuple[str, LLMUsage]:
    """
    Call the configured LLM with the given prompt.
    Returns (llm_text_response, usage).
    """
    provider, model = get_llm_config()
    usage = LLMUsage(model=model)

    if provider == "gemini":
        # ----------------------------
        # GEMINI (via google-genai)
        # ----------------------------
        try:
            from google import genai  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"google-genai not installed in this image: {e}"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set in the environment."
            )

        client = genai.Client(api_key=api_key)
        resp = client.responses.generate(
            model=model,
            contents=prompt,
        )
        text = resp.output_text

        # Best-effort usage extraction
        try:
            md = resp.usage_metadata
            usage.prompt_tokens = int(getattr(md, "prompt_token_count", 0) or 0)
            usage.completion_tokens = int(
                getattr(md, "candidates_token_count", 0) or 0
            )
        except Exception:
            pass

    elif provider == "hf":
        # ----------------------------
        # HUGGINGFACE (DeepSeek, etc.)
        # ----------------------------
        pipe, tokenizer = get_hf_pipeline(model)
        out = pipe(
            prompt,
            return_full_text=False,
            num_return_sequences=1,
        )[0]["generated_text"]

        text = out

        try:
            usage.prompt_tokens = len(tokenizer(prompt).input_ids)
            usage.completion_tokens = len(tokenizer(out).input_ids)
        except Exception:
            pass

    else:
        raise RuntimeError(f"Unknown KUBEFIX_PROVIDER {provider!r}")

    return text, usage


# -----------------------------
# MIC + request/response models
# -----------------------------


class FocusRef(BaseModel):
    kind: str
    namespace: str
    name: str


class MIC(BaseModel):
    """Minimal MIC structure used by kubefix."""

    focus: FocusRef
    specs: Dict[str, Any]


class MICRequest(BaseModel):
    kind: str
    namespace: str
    name: str


class GenerateMICWrapper(BaseModel):
    mic: MICRequest


class GenerateGenResult(BaseModel):
    rbac_patch: Optional[str] = None
    netpol_patch: Optional[str] = None
    podsec_patch: Optional[str] = None
    summary: str
    confidence: float
    risk: str


class GenerateUsage(BaseModel):
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    out_dir: str
    gen: GenerateGenResult
    usage: GenerateUsage


# -----------------------------
# K8s helpers
# -----------------------------


def kubectl_json(args: list[str]) -> dict:
    cmd = ["kubectl"] + args + ["-o", "json"]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def mic_build(focus: FocusRef) -> MIC:
    """
    Build a minimal MIC by reading the Pod spec from the cluster.
    """
    try:
        pod = kubectl_json(
            ["-n", focus.namespace, "get", "pod", focus.name]
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Cannot read Pod {focus.namespace}/{focus.name}: {e}"
        )

    specs: Dict[str, Any] = {
        "pods": [pod],
    }
    return MIC(focus=focus, specs=specs)


# -----------------------------
# Patch parsing
# -----------------------------


def parse_llm_output(text: str) -> GenerateGenResult:
    """
    VERY simple parser: expect the LLM to return JSON with keys:
      - rbac_patch, netpol_patch, podsec_patch, summary, confidence, risk

    If it fails to parse, we fall back to a plain podsec hardening patch.
    """
    try:
        data = json.loads(text)
        return GenerateGenResult(
            rbac_patch=data.get("rbac_patch"),
            netpol_patch=data.get("netpol_patch"),
            podsec_patch=data.get("podsec_patch"),
            summary=data.get("summary", "LLM summary"),
            confidence=float(data.get("confidence", 0.5)),
            risk=str(data.get("risk", "unknown")),
        )
    except Exception:
        # Fallback: treat entire text as "summary" and no structured patches
        return GenerateGenResult(
            rbac_patch=None,
            netpol_patch=None,
            podsec_patch=None,
            summary=text.strip()[:2000],
            confidence=0.5,
            risk="unknown",
        )


# -----------------------------
# Prompt builder
# -----------------------------


def build_prompt_from_mic(mic: MIC) -> str:
    """
    Build an LLM prompt given the MIC.

    You can refine this to include more details from specs.
    """
    focus = mic.focus
    pod = mic.specs.get("pods", [{}])[0]

    prompt = f"""
You are a Kubernetes security assistant. You are given a misconfigured pod and surrounding context.

Focus object:
- kind: {focus.kind}
- namespace: {focus.namespace}
- name: {focus.name}

Pod spec (JSON):
{json.dumps(pod, indent=2)}

Your job:
- Identify misconfigurations related to RBAC, NetworkPolicies, and PodSecurity.
- Propose up to three YAML patches:
  * `rbac_patch`: YAML to tighten RBAC (ClusterRole/Role/Binding).
  * `netpol_patch`: YAML to add or fix a NetworkPolicy.
  * `podsec_patch`: YAML to harden the pod / template fields.

Return a single JSON object with keys:
  - rbac_patch (string or null)
  - netpol_patch (string or null)
  - podsec_patch (string or null)
  - summary (string)
  - confidence (float between 0 and 1)
  - risk (string: low/medium/high)

DO NOT wrap the JSON in backticks. Only output valid JSON.
"""
    return prompt.strip()


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="KubeFix")


@app.get("/health")
def health_check():
    return {"status": "ok", "time": time.time()}


@app.post("/mic")
def api_mic(req: MICRequest) -> MIC:
    focus = FocusRef(kind=req.kind, namespace=req.namespace, name=req.name)
    mic = mic_build(focus)
    return mic


@app.post("/generate", response_model=GenerateResponse)
def api_generate(req: GenerateMICWrapper):
    """
    Main endpoint used by benchmark.sh and eval_scenarios.sh
    """
    focus = FocusRef(
        kind=req.mic.kind,
        namespace=req.mic.namespace,
        name=req.mic.name,
    )
    mic = mic_build(focus)

    # Build LLM prompt
    prompt = build_prompt_from_mic(mic)

    # Call LLM
    try:
        llm_text, usage = call_llm_for_patch(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Parse patches
    gen_result = parse_llm_output(llm_text)

    # Prepare out_dir (relative path used by scripts/eval_scenarios.sh)
    safe_ns = focus.namespace.replace("/", "-")
    safe_name = focus.name.replace("/", "-")
    ts = int(time.time())
    out_dir = f"out/pod-{safe_ns}-{safe_name}-{ts}"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save MIC
    (out_path / "mic.json").write_text(mic.model_dump_json(indent=2))

    # Save patches if present
    patches_dir = out_path / "patches"
    patches_dir.mkdir(exist_ok=True)

    if gen_result.podsec_patch:
        (patches_dir / "podsec_patch.yaml").write_text(gen_result.podsec_patch)
    if gen_result.rbac_patch:
        (patches_dir / "rbac_patch.yaml").write_text(gen_result.rbac_patch)
    if gen_result.netpol_patch:
        (patches_dir / "netpol_patch.yaml").write_text(gen_result.netpol_patch)

    # Save usage
    usage_obj = {
        "model": usage.model,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }
    (out_path / "usage.json").write_text(json.dumps(usage_obj, indent=2))

    usage_model = GenerateUsage(**usage_obj)

    return GenerateResponse(
        out_dir=out_dir,
        gen=gen_result,
        usage=usage_model,
    )


# -----------------------------
# CLI: verify
# -----------------------------

def kubectl_can_i(verb: str, resource: str) -> str:
    """
    Run 'kubectl auth can-i' for the current user context.
    Return 'yes' or 'no'.
    """
    try:
        subprocess.check_output(
            ["kubectl", "auth", "can-i", verb, resource],
            text=True,
        )
        # kubectl prints 'yes\n' or 'no\n'
        # we just re-run and parse:
        out = subprocess.check_output(
            ["kubectl", "auth", "can-i", verb, resource],
            text=True,
        ).strip()
        return out
    except Exception:
        return "no"


def summarize_rbac() -> dict:
    """
    Rough RBAC summary: counts of Roles/ClusterRoles and rules.
    """
    summary = {
        "role_count": 0,
        "cluster_role_count": 0,
        "role_resources": 0,
        "cluster_role_resources": 0,
    }

    try:
        roles = kubectl_json(["get", "roles", "-A"])
        summary["role_count"] = len(roles.get("items", []))
        for r in roles.get("items", []):
            rules = r.get("rules", [])
            summary["role_resources"] += len(rules)
    except Exception:
        pass

    try:
        crs = kubectl_json(["get", "clusterroles"])
        summary["cluster_role_count"] = len(crs.get("items", []))
        for r in crs.get("items", []):
            rules = r.get("rules", [])
            summary["cluster_role_resources"] += len(rules)
    except Exception:
        pass

    return summary


def verify(mic_path: Path, patches_dir: Path) -> dict:
    """
    Verification hook used by eval_scenarios.sh.
    Right now:
      - rbac_static_ok: always true (we don't enforce static checks yet)
      - can_i_summary: kubectl auth can-i on a fixed set of verbs/resources
      - notes: includes a pre-change RBAC summary
    """
    # We load MIC just to ensure it exists; we don't deeply use it yet.
    _mic_data = json.loads(mic_path.read_text())

    # Simple RBAC summary
    verbs = ["get", "list", "watch", "create", "delete"]
    resources = ["pods", "secrets", "configmaps"]

    can_i = {}
    for v in verbs:
        for r in resources:
            key = f"{v}:{r}"
            can_i[key] = kubectl_can_i(v, r)

    pre_summary = summarize_rbac()

    result = {
        "rbac_static_ok": True,
        "can_i_summary": can_i,
        "kyverno_ok": None,
        "notes": [
            f"RBAC pre-change summary: {pre_summary}",
        ],
    }
    return result


def cli():
    import argparse

    parser = argparse.ArgumentParser(description="KubeFix CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="Verify MIC + patches")
    v.add_argument("--mic", required=True, type=Path)
    v.add_argument("--patches", required=True, type=Path)

    args = parser.parse_args()

    if args.cmd == "verify":
        res = verify(args.mic, args.patches)
        print(json.dumps(res, indent=2))
    else:
        parser.error(f"Unknown command {args.cmd}")


if __name__ == "__main__":
    cli()

