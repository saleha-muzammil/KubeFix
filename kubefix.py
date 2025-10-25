# ─────────────────────────────────────────────────────────────────────────────
# KubeFix: Week 1–3 Starter Code
# Layout: single file for convenience; split into modules later if you prefer.
# Requires: Python 3.10+, kubectl context to a dev/test cluster, Gemini API key.
#
# Quickstart:
#   1) python kubefix.py mic --focus kind=Pod ns=default name=nginx-abc
#   2) python kubefix.py gen --mic mic.json --out out/
#   3) python kubefix.py verify --mic mic.json --patches out/patches
#   4) python kubefix.py serve  # Knative/Cloud Run friendly FastAPI
#
# Env:
#   export GEMINI_API_KEY=...            (Google AI Studio key)
#   export GEMINI_MODEL=gemini-2.0-flash (or gemini-2.0-pro)
#
# Optional CLIs used by verifier (install as needed):
#   - kubectl (must be on PATH)
#   - kyverno (for local policy tests)
#   - rbac-lookup (optional; enhances RBAC explanation)
#
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os, json, sys, subprocess, shlex, time, tempfile, base64
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import datetime as dt

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Lazy import Kubernetes client so mic can run without it installed (for CI)
K8S_AVAILABLE = True
try:
    from kubernetes import client, config
    from kubernetes.client import ApiException
except Exception:
    K8S_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: str, check: bool = True, capture: bool = True, env: Optional[dict]=None, cwd: Optional[str]=None):
    """Run a shell command and return (rc, out, err)."""
    p = subprocess.run(shlex.split(cmd), capture_output=capture, text=True, env=env, cwd=cwd)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p.returncode, p.stdout, p.stderr


def mkdirp(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o):
    """Robust JSON fallback for datetimes and odd k8s client models/enums."""
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    try:
        return str(o)
    except Exception:
        return "<unserializable>"

# ─────────────────────────────────────────────────────────────────────────────
# MIC (Minimal Incident Capsule) – Week 1
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FocusRef:
    kind: str
    namespace: str
    name: str

@dataclass
class MIC:
    focus: FocusRef
    owners: List[Dict[str, Any]]
    rbac: Dict[str, Any]
    network: Dict[str, Any]
    specs: Dict[str, Any]


def k8s_load():
    if not K8S_AVAILABLE:
        raise RuntimeError("kubernetes client not installed. pip install kubernetes")
    try:
        config.load_kube_config()
    except Exception:
        # Inside a pod
        config.load_incluster_config()


def get_api():
    core = client.CoreV1Api()
    apps = client.AppsV1Api()
    rbac_api = client.RbacAuthorizationV1Api()
    net_api = client.NetworkingV1Api()
    return core, apps, rbac_api, net_api


def mic_build(focus: FocusRef) -> MIC:
    """Collect just-enough context around the focus object for safe patching."""
    k8s_load()
    core, apps, rbac_api, net_api = get_api()

    owners: List[Dict[str, Any]] = []
    rbac: Dict[str, Any] = {"serviceAccount": None, "roleBindings": [], "roles": [], "clusterRoleBindings": [], "clusterRoles": []}
    network: Dict[str, Any] = {"services": [], "endpoints": [], "networkPolicies": []}
    specs: Dict[str, Any] = {}

    # 1) Resolve focus → Pod (we allow focus kind to be Pod/Deployment/StatefulSet)
    pod_list = []
    if focus.kind.lower() == "pod":
        try:
            pod = core.read_namespaced_pod(name=focus.name, namespace=focus.namespace)
            pod_list = [pod]
        except ApiException as e:
            raise RuntimeError(f"Cannot read Pod {focus.namespace}/{focus.name}: {e}")
    elif focus.kind.lower() in ("deployment", "statefulset"):
        if focus.kind.lower() == "deployment":
            obj = apps.read_namespaced_deployment(focus.name, focus.namespace)
            selector = obj.spec.selector.match_labels or {}
        else:
            obj = apps.read_namespaced_stateful_set(focus.name, focus.namespace)
            selector = obj.spec.selector.match_labels or {}
        lbl = ",".join([f"{k}={v}" for k, v in selector.items()])
        pod_list = core.list_namespaced_pod(namespace=focus.namespace, label_selector=lbl).items
        owners.append({"kind": focus.kind, "ns": focus.namespace, "name": focus.name, "selector": selector})
    else:
        raise RuntimeError("Focus kind must be one of: Pod, Deployment, StatefulSet")

    if not pod_list:
        raise RuntimeError("No pods resolved for focus.")

    # 2) Ownership chain (Pod → ReplicaSet → Deployment)
    for p in pod_list:
        md = p.metadata.owner_references or []
        specs.setdefault("pods", []).append(p.to_dict())
        owners.append({"kind": "Pod", "ns": p.metadata.namespace, "name": p.metadata.name})
        for o in md:
            owners.append({"kind": o.kind, "ns": p.metadata.namespace, "name": o.name})

    # 3) ServiceAccount + RBAC
    sa_names = sorted(set([(p.spec.service_account_name or "default", p.metadata.namespace) for p in pod_list]))
    if sa_names:
        sa_name, sa_ns = sa_names[0]
        try:
            sa = core.read_namespaced_service_account(sa_name, sa_ns)
            rbac["serviceAccount"] = sa.to_dict()
        except ApiException:
            pass

    # Gather RoleBindings / ClusterRoleBindings that reference this SA
    try:
        rbs = client.RbacAuthorizationV1Api().list_namespaced_role_binding(sa_ns)
        for rb in rbs.items:
            subjects = rb.subjects or []
            if any(s.kind == "ServiceAccount" and s.name == sa_name and (s.namespace or sa_ns) == sa_ns for s in subjects):
                rbac["roleBindings"].append(rb.to_dict())
                if rb.role_ref.kind == "Role":
                    try:
                        role = client.RbacAuthorizationV1Api().read_namespaced_role(rb.role_ref.name, sa_ns)
                        rbac["roles"].append(role.to_dict())
                    except ApiException:
                        pass
                elif rb.role_ref.kind == "ClusterRole":
                    try:
                        cr = client.RbacAuthorizationV1Api().read_cluster_role(rb.role_ref.name)
                        rbac["clusterRoles"].append(cr.to_dict())
                    except ApiException:
                        pass
    except ApiException:
        pass

    try:
        crbs = client.RbacAuthorizationV1Api().list_cluster_role_binding()
        for crb in crbs.items:
            subjects = crb.subjects or []
            if any(s.kind == "ServiceAccount" and s.name == sa_name and (s.namespace or sa_ns) == sa_ns for s in subjects):
                rbac["clusterRoleBindings"].append(crb.to_dict())
                if crb.role_ref.kind == "ClusterRole":
                    try:
                        cr = client.RbacAuthorizationV1Api().read_cluster_role(crb.role_ref.name)
                        rbac["clusterRoles"].append(cr.to_dict())
                    except ApiException:
                        pass
    except ApiException:
        pass

    # 4) Networking: Services/Endpoints/NetworkPolicies that select these pods
    all_labels = {}
    for p in pod_list:
        for k, v in (p.metadata.labels or {}).items():
            all_labels.setdefault(k, set()).add(v)

    # Services
    svcs = core.list_namespaced_service(focus.namespace).items
    for s in svcs:
        sel = (s.spec.selector or {})
        if sel and all(k in all_labels and s.spec.selector[k] in all_labels[k] for k in sel.keys()):
            network["services"].append(s.to_dict())
            try:
                ep = core.read_namespaced_endpoints(s.metadata.name, s.metadata.namespace)
                network["endpoints"].append(ep.to_dict())
            except ApiException:
                pass

    # NetworkPolicies
    try:
        nps = net_api.list_namespaced_network_policy(focus.namespace).items
        for np in nps:
            network["networkPolicies"].append(np.to_dict())
    except ApiException:
        pass

    return MIC(
        focus=focus,
        owners=owners,
        rbac=rbac,
        network=network,
        specs=specs,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Gemini SLM Client + Patch Generation – Week 1–2
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "rbac_patch": {"type": ["string", "null"], "description": "Strategic-merge or JSONPatch YAML"},
        "netpol_patch": {"type": ["string", "null"]},
        "podsec_patch": {"type": ["string", "null"]},
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "risk": {"type": "string"}
    },
    "required": ["summary", "confidence", "risk"]
}

GEMINI_SYSTEM_RULES = (
    "You are KubeFix, a Kubernetes security patching copilot.\n"
    "Operate ONLY within the provided MIC JSON. Produce minimal, least-privilege changes.\n"
    "Prefer removing wildcards in Roles, tightening verbs/resources;\n"
    "Prefer additive NetworkPolicies that preserve currently-allowed traffic;\n"
    "Harden pods per Pod Security Standards (baseline/restricted) without breaking mounts or required caps.\n"
    "Output ONLY JSON. Do not include prose outside the JSON.\n"
    "If a patch is not needed, set that field to null (not empty string). "
    "All patch fields must be valid Kubernetes YAML as strings."
)

def gemini_generate_patch(mic: MIC, model: Optional[str]=None) -> Dict[str, Any]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    url = GEMINI_API.format(model=model)

    schema_json = json.dumps(PATCH_SCHEMA, indent=2)
    mic_json = json.dumps(asdict(mic), default=_json_default, indent=2)

    payload = {
        "contents": [
            {"role": "model", "parts": [{"text": GEMINI_SYSTEM_RULES}]},
            {"role": "user", "parts": [
                {"text": "MIC JSON:"},
                {"text": mic_json},
                {"text": (
                    "Return a SINGLE JSON object with fields exactly:\n"
                    "rbac_patch (string|null), netpol_patch (string|null), podsec_patch (string|null), "
                    "summary (string), confidence (number 0..1), risk (string).\n"
                    "Each *patch* field must contain Kubernetes YAML (string). "
                    "If you propose a Role/ClusterRole change, make it a valid Role/ClusterRole resource. "
                    "For Pod hardening, prefer patching the Deployment's podTemplate; otherwise use JSONPatch for the live Pod. "
                    "If a patch is not needed, use null.\n\n"
                    f"Schema for reference:\n{schema_json}"
                )},
            ]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    r = requests.post(url, params={"key": api_key}, json=payload, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"Gemini API error: {r.status_code} {r.text}")

    data = r.json()

    # Try strict JSON parse from the single part
    text_block = None
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text_block = "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        text_block = None

    parsed: Dict[str, Any] = {}
    if text_block:
        try:
            parsed = json.loads(text_block)
        except Exception:
            parsed = {}

    # Fallback to old heuristic if provider ignored responseMimeType
    if not parsed:
        result = {"rbac_patch": None, "netpol_patch": None, "podsec_patch": None, "summary": "", "confidence": 0.0, "risk": "unknown"}
        try:
            # Extract fenced code blocks + trailing JSON
            import re
            big = text_block or ("\n".join(p.get("text", "") for p in parts) if 'parts' in locals() else "")
            def grab(label):
                m = re.search(rf'{label}.*?```(?:yaml|yml|json|JSON)?\n(.*?)```', big, re.I|re.S)
                return m.group(1).strip() if m else None
            result["rbac_patch"]   = grab("rbac_patch")
            result["netpol_patch"] = grab("netpol_patch")
            result["podsec_patch"] = grab("podsec_patch")
            jm = re.search(r"```json\n(\{[\s\S]*?\})\n```", big)
            if jm:
                meta = json.loads(jm.group(1))
                for k in ["summary","confidence","risk"]:
                    if k in meta:
                        result[k] = meta[k]
        except Exception:
            pass
        parsed = result

    # Normalize: empty strings → None; trim whitespace
    for k in ["rbac_patch", "netpol_patch", "podsec_patch"]:
        v = parsed.get(k)
        if isinstance(v, str):
            v2 = v.strip()
            parsed[k] = v2 if v2 else None

    parsed.setdefault("summary", "")
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("risk", "unknown")

    return parsed


def write_patch_report(out_dir: Path, mic: MIC, gen: Dict[str, Any]):
    mkdirp(out_dir / "patches")
    (out_dir / "mic.json").write_text(json.dumps(asdict(mic), default=_json_default, indent=2))
    (out_dir / "report.json").write_text(json.dumps(gen, default=_json_default, indent=2))
    for key in ("rbac_patch", "netpol_patch", "podsec_patch"):
        val = gen.get(key)
        if isinstance(val, str) and val.strip():
            (out_dir / "patches" / f"{key}.yaml").write_text(val.strip())

# ─────────────────────────────────────────────────────────────────────────────
# Verification – Week 2–3
# ─────────────────────────────────────────────────────────────────────────────

class VerifyResult(BaseModel):
    rbac_static_ok: bool
    can_i_summary: Dict[str, Any]
    kyverno_ok: Optional[bool] = None
    notes: List[str] = []


def rbac_static_diff(mic: MIC) -> Dict[str, Any]:
    """Compute a simple diff of Role/ClusterRole rules (pre-apply view from MIC)."""
    def rules_of(role_obj):
        try:
            return [r.to_dict() for r in role_obj.rules]
        except Exception:
            return role_obj.get("rules", [])

    roles = mic.rbac.get("roles", [])
    croles = mic.rbac.get("clusterRoles", [])
    return {
        "role_count": len(roles),
        "cluster_role_count": len(croles),
        "role_resources": sum(len(rules_of(r)) for r in roles),
        "cluster_role_resources": sum(len(rules_of(r)) for r in croles),
    }


def kubectl_can_i(ns: str, sa: str, probes: Optional[List[List[str]]] = None) -> Dict[str, Any]:
    """Run a small battery of `kubectl auth can-i` checks under SA identity."""
    if probes is None:
        probes = [
            ["get", "pods"], ["list", "pods"], ["watch", "pods"],
            ["get", "secrets"], ["list", "secrets"],
            ["get", "configmaps"], ["list", "configmaps"],
            ["create", "pods"], ["delete", "pods"],
        ]
    summary = {}
    for verb, resource in probes:
        cmd = f"kubectl auth can-i {verb} {resource} --as=system:serviceaccount:{ns}:{sa} --namespace {ns}"
        _, out, _ = run(cmd, check=False)
        summary[f"{verb}:{resource}"] = out.strip()
    return summary


def kyverno_test(patch_dir: Path, mic_file: Path) -> Optional[bool]:
    """If kyverno CLI exists, do a basic `kyverno apply` dry-run with patched resources.
    Users can add organization policies via KUBEFIX_KYVERNO_POLICY_DIR env var.
    """
    if shutil_which("kyverno") is None:
        return None
    policy_dir = os.environ.get("KUBEFIX_KYVERNO_POLICY_DIR")
    tmp = Path(tempfile.mkdtemp(prefix="kubefix-"))
    mic = json.loads(Path(mic_file).read_text())
    manifests = []
    import yaml

    # Pods
    for p in mic.get("specs", {}).get("pods", []):
        manifests.append(p)
    # RBAC
    manifests.extend(mic.get("rbac", {}).get("roles", []))
    manifests.extend(mic.get("rbac", {}).get("clusterRoles", []))
    # NetworkPolicies
    manifests.extend(mic.get("network", {}).get("networkPolicies", []))

    (tmp / "mic.yaml").write_text("\n---\n".join(yaml.safe_dump(m) for m in manifests))

    # Apply patches logically by concatenation (kyverno evaluate against the union)
    patch_texts = []
    for f in sorted((patch_dir).glob("*.yaml")):
        patch_texts.append(f.read_text())
    (tmp / "patches.yaml").write_text("\n---\n".join(patch_texts))

    cmd = f"kyverno apply {(policy_dir or '')} -r {tmp/'mic.yaml'} -r {tmp/'patches.yaml'}"
    rc, out, err = run(cmd, check=False)
    return rc == 0


def shutil_which(x: str) -> Optional[str]:
    from shutil import which
    return which(x)


def verify(mic_path: Path, patches_dir: Path) -> VerifyResult:
    mic = MIC(**json.loads(mic_path.read_text()))
    notes: List[str] = []

    # Static summary
    static = rbac_static_diff(mic)
    notes.append(f"RBAC pre-change summary: {static}")

    # can-i checks
    sa = mic.rbac.get("serviceAccount") or {}
    sa_name = (sa.get("metadata", {}) or {}).get("name", "default")
    sa_ns = (sa.get("metadata", {}) or {}).get("namespace", mic.focus.namespace)
    cani = kubectl_can_i(sa_ns, sa_name)

    # kyverno local test (optional)
    ky = kyverno_test(patches_dir, mic_path)

    return VerifyResult(
        rbac_static_ok=True,
        can_i_summary=cani,
        kyverno_ok=ky,
        notes=notes,
    )

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI wrapper – Week 3 (Knative/Cloud Run)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="KubeFix")

class MICRequest(BaseModel):
    kind: str
    namespace: str
    name: str

class GenerateRequest(BaseModel):
    mic: MICRequest

@app.post("/mic")
async def api_mic(req: MICRequest):
    mic = mic_build(FocusRef(kind=req.kind, namespace=req.namespace, name=req.name))
    return asdict(mic)

@app.post("/generate")
async def api_generate(req: GenerateRequest):
    mic = mic_build(FocusRef(kind=req.mic.kind, namespace=req.mic.namespace, name=req.mic.name))
    gen = gemini_generate_patch(mic)
    out = Path("out") / f"{mic.focus.kind.lower()}-{mic.focus.namespace}-{mic.focus.name}-{int(time.time())}"
    write_patch_report(out, mic, gen)
    return {"out_dir": str(out), "gen": gen}

# NEW: Falco webhook endpoint → MIC → patch generation
@app.post("/falco")
async def falco_webhook(event: Dict[str, Any]):
    of = (event.get("output_fields") or {})
    ns = of.get("k8s.ns.name") or of.get("k8s_namespace_name") or of.get("k8s.ns") or of.get("k8s_namespace")
    pod = of.get("k8s.pod.name") or of.get("k8s_pod_name") or of.get("k8s.pod") or of.get("k8s_pod")
    if not pod:
        raise HTTPException(status_code=400, detail="Falco event missing k8s pod name")
    if not ns:
        ns = "default"
    try:
        mic = mic_build(FocusRef(kind="Pod", namespace=ns, name=pod))
        gen = gemini_generate_patch(mic)
        out = Path("out") / f"falco-{ns}-{pod}-{int(time.time())}"
        write_patch_report(out, mic, gen)
        return {"ok": True, "out_dir": str(out), "confidence": gen.get("confidence", 0), "risk": gen.get("risk", "unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW: Kubernetes Audit webhook → MIC → patch generation
@app.post("/audit")
async def audit_webhook(payload: Dict[str, Any]):
    """
    Accepts Kubernetes Audit v1 payloads. Handles both single-event and
    event-list formats. Attempt to resolve a Pod focus from higher-level refs.
    """
    events: List[Dict[str, Any]] = []
    if payload.get("kind") == "EventList" and payload.get("items"):
        events = payload["items"]
    else:
        events = [payload]

    results = []
    for ev in events:
        objref = ev.get("objectRef", {})
        kind = (objref.get("resource") or objref.get("kind") or "").lower()
        ns = objref.get("namespace") or "default"
        name = objref.get("name")

        focus_kind = None
        focus_name = None
        focus_ns = ns

        try:
            if kind in ("pods", "pod"):
                focus_kind = "Pod"
                focus_name = name
            elif kind in ("deployments", "deployment"):
                k8s_load(); core, apps, *_ = get_api()
                dep = apps.read_namespaced_deployment(name, ns)
                selector = dep.spec.selector.match_labels or {}
                lbl = ",".join([f"{k}={v}" for k, v in selector.items()])
                pods = core.list_namespaced_pod(ns, label_selector=lbl).items
                if pods:
                    focus_kind = "Pod"
                    focus_name = pods[0].metadata.name
            elif kind in ("replicasets", "replicaset"):
                k8s_load(); core, apps, *_ = get_api()
                rs = apps.read_namespaced_replica_set(name, ns)
                selector = rs.spec.selector.match_labels or {}
                lbl = ",".join([f"{k}={v}" for k, v in selector.items()])
                pods = core.list_namespaced_pod(ns, label_selector=lbl).items
                if pods:
                    focus_kind = "Pod"
                    focus_name = pods[0].metadata.name
            if not focus_name:
                ro = ev.get("requestObject") or {}
                tmpl = (((ro.get("spec") or {}).get("template") or {}).get("metadata") or {}).get("labels") or {}
                if tmpl:
                    k8s_load(); core, *_ = get_api()
                    lbl = ",".join([f"{k}={v}" for k, v in tmpl.items()])
                    pods = core.list_namespaced_pod(ns, label_selector=lbl).items
                    if pods:
                        focus_kind = "Pod"; focus_name = pods[0].metadata.name

            if focus_kind and focus_name:
                mic = mic_build(FocusRef(kind=focus_kind, namespace=focus_ns, name=focus_name))
                gen = gemini_generate_patch(mic)
                out = Path("out") / f"audit-{focus_ns}-{focus_name}-{int(time.time())}"
                write_patch_report(out, mic, gen)
                results.append({"ok": True, "ns": focus_ns, "pod": focus_name, "out_dir": str(out), "confidence": gen.get("confidence", 0)})
            else:
                results.append({"ok": False, "reason": "Could not resolve a Pod from audit event", "objref": objref})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {"results": results}

# ─────────────────────────────────────────────────────────────────────────────
# CLI – glue for local runs
# ─────────────────────────────────────────────────────────────────────────────

def cli():
    import argparse
    ap = argparse.ArgumentParser("kubefix")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_mic = sub.add_parser("mic")
    s_mic.add_argument("--focus", required=True, help="kind=Pod ns=default name=nginx-xyz")
    s_mic.add_argument("--out", default="mic.json")

    s_gen = sub.add_parser("gen")
    s_gen.add_argument("--mic", required=True)
    s_gen.add_argument("--out", default="out")

    s_ver = sub.add_parser("verify")
    s_ver.add_argument("--mic", required=True)
    s_ver.add_argument("--patches", required=True)

    s_srv = sub.add_parser("serve")
    s_srv.add_argument("--port", type=int, default=8080)

    args = ap.parse_args()

    if args.cmd == "mic":
        kv = dict(p.split("=", 1) for p in args.focus.split())
        focus = FocusRef(kind=kv["kind"], namespace=kv["ns"], name=kv["name"])
        mic = mic_build(focus)
        Path(args.out).write_text(json.dumps(asdict(mic), default=_json_default, indent=2))
        print(f"Wrote {args.out}")

    elif args.cmd == "gen":
        mic = MIC(**json.loads(Path(args.mic).read_text()))
        gen = gemini_generate_patch(mic)
        out_dir = Path(args.out)
        write_patch_report(out_dir, mic, gen)
        print(f"Patch report written to {out_dir}")

    elif args.cmd == "verify":
        res = verify(Path(args.mic), Path(args.patches))
        print(res.json(indent=2))

    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    cli()

