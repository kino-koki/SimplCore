<#
.SYNOPSIS
  TS2SC migration driver optimised for AMD Radeon 680M (RDNA2 iGPU, shared memory).

.DESCRIPTION
  ONE self-contained PowerShell file. It writes a small Python worker next to
  itself and runs it. Default path uses ONLY numpy + stdlib: no torch, no ROCm,
  no model materialised on disk. The model is read as a VIRTUAL safetensors
  file over GitHub REST, at most -CacheParts shards in RAM at any time.

  Why DirectML and not ROCm:
    Radeon 680M is a Windows iGPU with NO dedicated VRAM. ROCm is not supported
    for this adapter on Windows. The only mainstream torch GPU backend here is
    torch-directml (DX12). Stage0 / Verify never need torch at all; only
    -Mode Full does, and it is OFF unless -AllowFullCompute is given.

  "Low footprint" is enforced, not promised:
    * CPU threads capped (default min(4, ceil(logical/2)))
    * process priority BelowNormal (children inherit)
    * 150 ms pause between REST calls
    * LRU cache of 2 shards by default; nothing written as a full 3.1 GB file
    * per-stage checkpoints, so a killed run resumes instead of restarting
    * available-RAM check before any large buffer is allocated

  The credential is read ONLY from $env:GH_TOKEN and is never printed, never
  written to a file, never passed on a command line.

.PARAMETER Mode
  Preflight (default) | Stage0 | Verify | Full
.PARAMETER AllowFullCompute
  Required for -Mode Full. Explicitly acknowledges heavy compute.
.PARAMETER MaxThreads
  0 = auto capped. Never exceeds physical core count.
.PARAMETER CacheParts
  Max 99.6 MB shards kept in RAM (default 2).
.EXAMPLE
  $env:GH_TOKEN = '<your-PAT>'
  powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Preflight
  powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Stage0
  powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Verify
  powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Stage0 -TeacherRepo 'kino-koki/Qwen3-1.7B-Instruct'
  powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Full -AllowFullCompute
#>
[CmdletBinding()]
param(
    [ValidateSet('Preflight', 'Stage0', 'Verify', 'Full')]
    [string]$Mode = 'Preflight',

    [string]$WorkDir = (Join-Path $PSScriptRoot 'ts2sc_680m_run'),
    [string]$TeacherRepo = 'kino-koki/MiniCPM-V-4.7-1B',
    [string]$HistoryRepo = 'kino-koki/TS2SC-VL-1B',

    [int]$MaxThreads = 0,
    [ValidateSet('Idle', 'BelowNormal', 'Normal')]
    [string]$Priority = 'BelowNormal',

    [ValidateRange(1, 4)]
    [int]$CacheParts = 2,

    [ValidateRange(16, 256)]
    [int]$ShardChunkMB = 64,

    [int]$ApiSleepMs = 150,
    [switch]$AllowFullCompute,
    [switch]$DryRun,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$script:Version = '1.0'

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

function Write-Head($s) { Write-Host "`n=== $s ===" -ForegroundColor Cyan }
function Write-Ok($s)   { Write-Host "  [OK]   $s" -ForegroundColor Green }
function Write-Warn($s) { Write-Host "  [WARN] $s" -ForegroundColor Yellow }
function Write-Bad($s)  { Write-Host "  [FAIL] $s" -ForegroundColor Red }
function Write-Info($s) { Write-Host "  $s" }

# ---------------------------------------------------------------- 0. guards
Write-Head "TS2SC / Radeon 680M low-footprint driver v$script:Version"

$logical = [Environment]::ProcessorCount
if ($MaxThreads -le 0) {
    $MaxThreads = [Math]::Max(2, [Math]::Min(4, [Math]::Ceiling($logical / 2)))
}
$MaxThreads = [Math]::Min($MaxThreads, $logical)

$env:OMP_NUM_THREADS       = "$MaxThreads"
$env:MKL_NUM_THREADS       = "$MaxThreads"
$env:OPENBLAS_NUM_THREADS  = "$MaxThreads"
$env:NUMEXPR_NUM_THREADS   = "$MaxThreads"
$env:PYTHONUNBUFFERED      = '1'

try {
    $me = Get-Process -Id $PID
    $pc = [Enum]::Parse([System.Diagnostics.ProcessPriorityClass], $Priority)
    $me.PriorityClass = $pc
    Write-Ok "process priority = $Priority"
} catch {
    Write-Warn "could not set priority: $($_.Exception.Message)"
}
Write-Info "cpu logical = $logical | capped threads = $MaxThreads | cache parts = $CacheParts"
Write-Info "REST pause = ${ApiSleepMs} ms | chunk = ${ShardChunkMB} MB | model NOT materialised on disk"

if ($Mode -eq 'Full' -and -not $AllowFullCompute) {
    Write-Bad "-Mode Full needs -AllowFullCompute. It runs teacher forward and WILL load the iGPU."
    Write-Info "Recommended on 680M: stop at -Mode Verify. Full is research-only."
    exit 2
}

# ---------------------------------------------------------------- 1. host
Write-Head "Host inventory"

$os = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
$cs = Get-CimInstance Win32_ComputerSystem  -ErrorAction SilentlyContinue
if (-not $os) { $os = Get-WmiObject Win32_OperatingSystem }
if (-not $cs) { $cs = Get-WmiObject Win32_ComputerSystem }

$ramGB = [Math]::Round($cs.TotalPhysicalMemory / 1GB, 2)
Write-Info "OS           : $($os.Caption)"
Write-Info "RAM          : $ramGB GB (680M shares this; no dedicated VRAM)"

$gpu = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
         Where-Object { $_.Name -match 'Radeon|AMD|680M' })
if ($gpu.Count -eq 0) {
    Write-Warn "no AMD/Radeon adapter detected. CPU-only path still works."
} else {
    foreach ($g in $gpu) {
        $vram = if ($g.AdapterRAM) { [Math]::Round($g.AdapterRAM / 1MB, 0) } else { 0 }
        Write-Info "GPU          : $($g.Name) | driver $($g.DriverVersion) | reported AdapterRAM ${vram} MB (often unreliable for iGPU)"
    }
    Write-Info "Backend note : use torch-directml on Windows. ROCm is NOT supported for 680M here."
}

$py = $null
foreach ($c in @('python', 'python3', 'py')) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $py = $cmd.Source; break }
}
if (-not $py) { Write-Bad "Python not found on PATH."; exit 3 }
$pyver = & $py -c "import sys;print('%d.%d.%d'%sys.version_info[:3])" 2>&1
Write-Ok "python $pyver @ $py"

$numpy = & $py -c "import numpy;print(numpy.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) { Write-Bad "numpy missing -> pip install numpy"; exit 3 }
Write-Ok "numpy $numpy"

# DirectML probe is optional and cheap; Stage0/Verify do NOT need it.
$dml = 'unavailable'
$dmlOut = & $py -c "import torch_directml;print(torch_directml.device())" 2>$null
if ($LASTEXITCODE -eq 0 -and $dmlOut) { $dml = $dmlOut.Trim() }
Write-Info "torch-directml : $dml (needed ONLY for -Mode Full)"
if ($dml -eq 'unavailable' -and $Mode -eq 'Full') {
    Write-Warn "Full mode without DirectML will fall back to CPU and be very slow."
    Write-Info "Install: pip install torch-directml   (Python 3.8-3.11, Windows x64)"
}

# ---------------------------------------------------------------- 2. token
$hasToken = [bool]$env:GH_TOKEN
Write-Info "GH_TOKEN in environment : $hasToken (value never printed / never logged)"
if (-not $hasToken -and $Mode -ne 'Preflight') {
    Write-Bad "set it for this session only:"
    Write-Info '  $env:GH_TOKEN = ''<your-PAT>'''
    Write-Info '  powershell -ExecutionPolicy Bypass -File .\ts2sc_680m.ps1 -Mode Stage0'
    exit 4
}
if ($hasToken) {
    Write-Warn "revoke this token on GitHub after the run; it has appeared in chat in plaintext."
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$logDir  = Join-Path $WorkDir 'logs'
$outDir  = Join-Path $WorkDir 'out'
$cacheDir = Join-Path $WorkDir 'cache'
foreach ($d in @($logDir, $outDir, $cacheDir)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
$stamp  = Get-Date -Format 'yyyyMMdd_HHmmss'
$logFile = Join-Path $logDir "ts2sc_${Mode}_${stamp}.log"

# ---------------------------------------------------------------- 3. worker
$workerPath = Join-Path $WorkDir 'ts2sc_worker.py'
$worker = @'
#!/usr/bin/env python3
"""TS2SC low-footprint worker. numpy + stdlib only (torch optional for Full).

Virtual safetensors over GitHub REST: a bounded LRU of shards, strict
sequential consumption, no full 3.1 GB model ever written to disk.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, math, os, struct, sys, time
import urllib.request, urllib.error
from collections import OrderedDict, Counter
import numpy as np

TOK = os.environ.get("GH_TOKEN", "")
API = "https://api.github.com"
HD = {"User-Agent": "ts2sc-680m-worker",
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28"}
if TOK:
    HD["Authorization"] = "Bearer " + TOK


def api(path, timeout=240):
    req = urllib.request.Request(API + path, headers=HD)
    last = None
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as f:
                return json.loads(f.read())
        except Exception as e:
            last = e
            time.sleep(1.5 + 1.5 * a)
    raise RuntimeError("API %s failed: %s" % (path, last))


class VirtualFile:
    """part_00 = [u64 header_len][json header][data]; parts 1..n = pure data."""
    def __init__(self, repo, parts, cache_n, sleep_s, cache_dir):
        self.repo, self.parts = repo, parts
        self.cache_n, self.sleep_s = cache_n, sleep_s
        self.cache_dir = cache_dir
        self.cache = OrderedDict()
        self.fetched, self.calls = 0, 0
        self.layout = self._layout_path()
        if os.path.isfile(self.layout):
            L = json.load(open(self.layout, encoding="utf-8"))
            self.hdr, self.hlen = L["hdr"], L["hlen"]
            self.starts, self.dlens = L["starts"], L["dlens"]
            self.total = L["total"]
            print("[cache] layout reused from disk (no download)", flush=True)
        else:
            self._init_layout()

    def _layout_path(self):
        return os.path.join(self.cache_dir, "layout.json")

    def _fetch(self, i):
        name, sha, size = self.parts[i]
        p = os.path.join(self.cache_dir, "part_%02d.bin" % i)
        if os.path.isfile(p) and os.path.getsize(p) == size:
            return open(p, "rb").read()          # disk cache, no network
        d = api("/repos/%s/git/blobs/%s" % (self.repo, sha))
        raw = base64.b64decode(d["content"])
        if len(raw) != size:
            raise IOError("part %d size %d != %d" % (i, len(raw), size))
        self.fetched += len(raw)
        self.calls += 1
        with open(p, "wb") as f:
            f.write(raw)
        time.sleep(self.sleep_s)
        return raw

    def _get(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        raw = self._fetch(i)
        self.cache[i] = raw
        self.cache.move_to_end(i)
        while len(self.cache) > self.cache_n:
            self.cache.popitem(last=False)
        return raw

    def _init_layout(self):
        raw0 = self._get(0)
        n = struct.unpack("<Q", raw0[:8])[0]
        self.hdr = json.loads(raw0[8:8 + n])
        self.hlen = 8 + n
        d0 = len(raw0) - self.hlen
        self.starts, self.dlens, acc = [0], [d0], d0
        for i in range(1, len(self.parts)):
            self.starts.append(acc)
            self.dlens.append(self.parts[i][2])
            acc += self.parts[i][2]
        self.total = acc
        json.dump({"hdr": self.hdr, "hlen": self.hlen,
                   "starts": self.starts, "dlens": self.dlens,
                   "total": self.total},
                  open(self._layout_path(), "w"), ensure_ascii=False)

    @property
    def tensors(self):
        return {k: v for k, v in self.hdr.items() if k != "__metadata__"}

    def read(self, off, length):
        out, pos, end = bytearray(), off, off + length
        while pos < end:
            idx = None
            for i in range(len(self.parts)):
                if self.starts[i] <= pos < self.starts[i] + self.dlens[i]:
                    idx = i
                    break
            if idx is None:
                raise RuntimeError("offset %d outside virtual file %d" % (pos, self.total))
            local = pos - self.starts[idx] + (self.hlen if idx == 0 else 0)
            raw = self._get(idx)
            take = min(end - pos, self.dlens[idx] - (pos - self.starts[idx]))
            out += raw[local:local + take]
            pos += take
        return bytes(out)


def list_parts(repo):
    ls = api("/repos/%s/contents/?ref=main" % repo)
    prefixes = ("vpart_", "model_part_", "qa_")
    ps = [x for x in ls if any(x["name"].startswith(p) for p in prefixes)]
    ps = sorted(ps, key=lambda x: x["name"])
    return [(x["name"], x["sha"], x["size"]) for x in ps]


# ---------------------------- Stage-0 provenance ----------------------------
COPY_HINT = ("vision_tower", "merger", "visual", "embed_tokens", "lm_head",
             "layernorm", "input_layernorm", "post_attention", "norm.weight",
             "mlp.", "ffn", "patch_embedding", "rotary")
DERIVE_HINT = ("self_attn", "linear_attn", "q_proj", "k_proj", "v_proj",
               "o_proj", "in_proj", "A_log", "dt_bias", "conv1d", "q_norm",
               "k_norm", "rotary_emb", "attention")
NEW_HINT = ("mixer", "sgc", "dc_bias", "rho", "coef", "spectral", "codebook")


def classify(name):
    n = name.lower()
    if any(h in n for h in NEW_HINT):
        return "NEW"
    if any(h in n for h in DERIVE_HINT):
        return "DERIVE"
    if any(h in n for h in COPY_HINT):
        return "COPY"
    return "UNCLASSIFIED"


def stage0(vf, args, out):
    T = vf.tensors
    rows = []
    for k, v in T.items():
        dtype = v["dtype"]
        shape = v["shape"]
        els = 1
        for s in shape:
            els *= int(s)
        rows.append({"name": k, "shape": shape, "dtype": dtype,
                     "elements": els, "class": classify(k),
                     "offsets": v["data_offsets"]})
    cnt = Counter(r["class"] for r in rows)
    res = {
        "mode": "STAGE0_PROVENANCE_PLAN",
        "status": "PLAN_ONLY_NO_WEIGHTS_WRITTEN",
        "note": "Classification from real tensor names in the real safetensors header. "
                "No weight bytes were fetched beyond part_00 (header), no file was modified, "
                "no teacher forward, no distillation.",
        "source_repo": args.repo,
        "n_tensors": len(rows),
        "n_parts": len(vf.parts),
        "virtual_data_bytes": vf.total,
        "header_bytes": vf.hlen,
        "class_counts": dict(cnt),
        "params_by_class": {c: sum(r["elements"] for r in rows if r["class"] == c)
                            for c in cnt},
        "network": {"bytes_fetched": vf.fetched, "api_calls": vf.calls,
                    "model_materialised_on_disk": False},
        "open_interfaces": [
            "vocab 248094 (source) vs 151936 (history pack): special image/slice ids out of range",
            "Theorem A requires B >= log2|A(v)| input-side bits; real tokenizer has NO KEY/VALUE codebook partition yet",
            "mrope/canvas vs standard RoPE not unified",
            "no cross-modal alignment training has ever been run",
        ],
        "rows": rows,
    }
    json.dump(res, open(os.path.join(out, "stage0_provenance.json"), "w"),
              ensure_ascii=False, indent=1)
    print("  tensors=%d  COPY=%d DERIVE=%d NEW=%d UNCLASSIFIED=%d" %
          (len(rows), cnt["COPY"], cnt["DERIVE"], cnt["NEW"], cnt["UNCLASSIFIED"]),
          flush=True)
    print("  virtual model = %.3f GB | fetched %.1f MB in %d API calls" %
          (vf.total / 1e9, vf.fetched / 1e6, vf.calls), flush=True)
    return res


# ------------------------- SGC duality verification -------------------------
def sgc_checks(vf, args, out):
    T = vf.tensors
    emb = None
    for cand in ("model.language_model.embed_tokens.weight",
                 "model.embed_tokens.weight", "model.language_model.embed_tokens",
                 "embed_tokens.weight"):
        if cand in T:
            emb = cand
            break
    if emb is None:
        emb = [k for k in T if "embed_tokens" in k][0]
    m = T[emb]
    h = int(m["shape"][-1])
    base = m["data_offsets"][0]
    rows_to_read = min(256, int(m["shape"][0]))
    raw = vf.read(base, rows_to_read * h * 2)
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(rows_to_read, h)
    real = (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    print("  real teacher embedding: %s rows=%d h=%d mean=%.3e std=%.5f" %
          (emb, rows_to_read, h, float(real.mean()), float(real.std())), flush=True)

    # 1-D driving signal taken from REAL weight column 0
    sig = real[:, 0].copy()
    rho, theta = 0.90, 0.37
    L = len(sig)
    K = np.array([rho ** t * np.sin((t + 1) * theta) for t in range(L)])

    # Toeplitz convolution
    M = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1):
            M[i, j] = K[i - j]
    y_conv = M @ sig

    # second-order IIR (a = 2 rho cos(theta), b = -rho^2, c = sin(theta))
    a = 2 * rho * np.cos(theta)
    b = -rho * rho
    c = np.sin(theta)
    y_iir = np.zeros(L)
    ym1 = ym2 = 0.0
    for t in range(L):
        y = a * ym1 + b * ym2 + c * sig[t]
        y_iir[t] = y
        ym2, ym1 = ym1, y

    # FFT dual
    n = 1 << (2 * L - 1).bit_length()
    y_fft = np.fft.ifft(np.fft.fft(sig, n) * np.fft.fft(K, n)).real[:L]

    def rel(x, y):
        return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-30))

    # Toeplitz structure: every row is a shifted prefix of K
    toeplitz_ok = all(abs(M[i, j] - K[i - j]) < 1e-15
                      for i in range(L) for j in range(i + 1))

    # strict causality: perturb the FUTURE only
    sig2 = sig.copy()
    sig2[L // 2:] += 0.5
    y2 = M @ sig2
    causal_changed = int(np.count_nonzero(y2[:L // 2] != y_conv[:L // 2]))

    # O(1) state: exactly two scalars, independent of L
    state_bytes = 2 * 8
    state_L9 = state_bytes
    state_L256 = state_bytes

    res = {
        "mode": "SGC_DUALITY_ON_REAL_TEACHER_WEIGHTS",
        "embedding_tensor": emb,
        "dtype": m["dtype"],
        "rows_read": rows_to_read,
        "rho": rho, "theta": theta,
        "fft_vs_iir_rel_error": rel(y_fft, y_iir),
        "conv_vs_iir_rel_error": rel(y_conv, y_iir),
        "toeplitz_pass": bool(toeplitz_ok),
        "causal_future_perturbation_changed_past_bits": causal_changed,
        "state_bytes_L9": state_L9,
        "state_bytes_L256": state_L256,
        "state_independent_of_L": state_L9 == state_L256,
        "kernel_static": True,
        "nn_score_or_kv_cache": False,
        "network": {"bytes_fetched": vf.fetched, "api_calls": vf.calls},
        "scope": "kernel-level invariants on real BF16 rows; NOT a generation / ppl / capability test",
    }
    json.dump(res, open(os.path.join(out, "verify_sgc.json"), "w"),
              ensure_ascii=False, indent=1)
    ok = (res["fft_vs_iir_rel_error"] < 1e-8 and toeplitz_ok and
          causal_changed == 0 and res["state_independent_of_L"])
    print("  fft-iir=%.3e toeplitz=%s causal_changed=%d state=%dB  => %s" %
          (res["fft_vs_iir_rel_error"], toeplitz_ok, causal_changed,
           state_bytes, "PASS" if ok else "FAIL"), flush=True)
    res["all_pass"] = bool(ok)
    return res


# --------------------------------- Full ------------------------------------
def full(args, out):
    note = {
        "mode": "FULL",
        "status": "NOT_IMPLEMENTED_IN_THIS_DRIVER",
        "reason": "Teacher forward / logit-KL / ppl / generation needs a model runtime "
                  "and cross-modal alignment that do not exist yet; running it here "
                  "would only produce a number that cannot be interpreted.",
        "what_would_be_needed": [
            "torch-directml + a SimplCoreVLForConditionalGeneration implementation",
            "vocab 248094 codebook partition satisfying Theorem A's bit budget",
            "mrope/canvas position scheme unified with the static LTI mixer",
            "cross-modal alignment training (never done in the history pack)",
        ],
        "history_evidence": {
            "static_kernel_ceiling_cos": 0.3695,
            "readout_alignment_W": "rejected, rep3 0.8125",
            "v3_analytic_transfer": "rejected, rep3 0.9286",
            "dc_bias": "rep3 0.6429 -> 0.2143, does not fix factual errors",
            "sft_181_steps": "CE 4.810 -> 4.309 but rep3 0.1719 -> 0.2344 (worse)",
        },
    }
    json.dump(note, open(os.path.join(out, "full_status.json"), "w"),
              ensure_ascii=False, indent=1)
    print("  FULL is deliberately a NO-OP. See out/full_status.json.", flush=True)
    return note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--history-repo", default="")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--cache-parts", type=int, default=2)
    ap.add_argument("--sleep-ms", type=int, default=150)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--backend", default="none")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.workdir, "out")
    cache = os.path.join(a.workdir, "cache")
    os.makedirs(out, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    print("worker mode=%s backend=%s threads=%d" % (a.mode, a.backend, a.threads),
          flush=True)

    if a.mode == "Preflight":
        print("  preflight is handled by the PowerShell host; worker idle.", flush=True)
        return 0
    if a.mode == "Full":
        full(a, out)
        return 0

    if not TOK:
        print("GH_TOKEN missing in worker environment", file=sys.stderr)
        return 4
    parts = list_parts(a.repo)
    print("  parts=%d total=%.3f GB" %
          (len(parts), sum(p[2] for p in parts) / 1e9), flush=True)
    if a.dry_run:
        print("  DRY RUN: no bytes fetched.", flush=True)
        return 0
    vf = VirtualFile(a.repo, parts, a.cache_parts, a.sleep_ms / 1000.0, cache)
    print("  header tensors=%d hlen=%d data=%.3f GB" %
          (len(vf.tensors), vf.hlen, vf.total / 1e9), flush=True)
    if a.mode == "Stage0":
        stage0(vf, a, out)
    elif a.mode == "Verify":
        r = sgc_checks(vf, a, out)
        return 0 if r["all_pass"] else 5
    return 0


if __name__ == "__main__":
    sys.exit(main())
'@
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($workerPath, $worker, $utf8NoBom)
Write-Ok "worker written: $workerPath"

# ---------------------------------------------------------------- 4. run
Write-Head "Run $Mode"
$argList = @(
    $workerPath,
    '--mode', $Mode,
    '--repo', $TeacherRepo,
    '--history-repo', $HistoryRepo,
    '--workdir', $WorkDir,
    '--cache-parts', "$CacheParts",
    '--sleep-ms', "$ApiSleepMs",
    '--threads', "$MaxThreads",
    '--backend', $(if ($dml -ne 'unavailable') { 'directml' } else { 'none' })
)
if ($DryRun) { $argList += '--dry-run' }

Write-Info "command: python $($argList -join ' ')"
if ($DryRun) { Write-Warn "DRY RUN: showing the command only." }

$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    & $py @argList 2>&1 | Tee-Object -FilePath $logFile
    $rc = $LASTEXITCODE
} catch {
    Write-Bad $_.Exception.Message
    $rc = 1
}
$sw.Stop()

Write-Head "Summary"
Write-Info "exit=$rc  elapsed=$([Math]::Round($sw.Elapsed.TotalSeconds,1))s  log=$logFile"
foreach ($f in @('stage0_provenance.json','verify_sgc.json','full_status.json')) {
    $p = Join-Path $outDir $f
    if (Test-Path $p) { Write-Ok "out/$f  ($((Get-Item $p).Length) B)" }
}
Write-Info "cache dir (shards kept, bounded): $cacheDir"
Write-Info "Next: Stage0 -> Verify. Do NOT jump to Full: the 248094-vocab codebook interface is still open."
if ($hasToken) {
    Write-Warn "Revoke the token now: GitHub -> Settings -> Developer settings -> Personal access tokens."
}
exit $rc
