#Requires -Version 7
# ========================================================================
# Qingxiaotuan LLM Turbo Distiller - Training Launcher (Windows PowerShell)
# ========================================================================
# Usage: powershell -File scripts\run_turbo.ps1 [options]
# ========================================================================

$ErrorActionPreference = "Stop"

# ---------- Console colors ----------
function Write-Color ([string]$text, [ConsoleColor]$color = [ConsoleColor]::Gray) {
    $old = [Console]::ForegroundColor
    try {
        [Console]::ForegroundColor = $color
        Write-Host $text
    } finally {
        [Console]::ForegroundColor = $old
    }
}
function Write-Info  ([string]$m) { Write-Host ("[{0}] [{1:HH:mm:ss}] {2}" -f "INFO ", (Get-Date), $m) -ForegroundColor Green }
function Write-Warn  ([string]$m) { Write-Host ("[{0}] [{1:HH:mm:ss}] {2}" -f "WARN ", (Get-Date), $m) -ForegroundColor Yellow }
function Write-Error ([string]$m) { Write-Host ("[{0}] [{1:HH:mm:ss}] {2}" -f "ERROR", (Get-Date), $m) -ForegroundColor Red }
function Write-Fatal ([string]$m, [int]$code = 1) { Write-Host ("[{0}] [{1:HH:mm:ss}] {2} (exit code $code)" -f "FATAL", (Get-Date), $m) -ForegroundColor Red; exit $code }

function Step-Start([int]$n, [string]$title) {
    Write-Host ""
    Write-Host ("=" * 72) -ForegroundColor Cyan
    Write-Host ("  Step ${n}: $title  (start: {0:yyyy-MM-dd HH:mm:ss})" -f (Get-Date)) -ForegroundColor Cyan
    Write-Host ("=" * 72) -ForegroundColor Cyan
}
function Step-Done([int]$n) {
    Write-Host ("-" * 72) -ForegroundColor Green
    Write-Host ("  Step ${n} done  (end: {0:yyyy-MM-dd HH:mm:ss})" -f (Get-Date)) -ForegroundColor Green
    Write-Host ("-" * 72) -ForegroundColor Green
}

# ---------- Argument parsing ----------
$PRESET = ""
$NUM_GPUS = 8
$HOURS = 24
$DATA_FILES = @()
$OUTPUT_DIR = "./output"
$SEQ_LEN = 2048
$MICRO_BATCH = 4
$GRAD_ACC = 8
$FP8 = "true"
$CPU_OFFLOAD = "false"
$RESUME_CKPT = ""
$DRY_RUN = "false"

$args = $args.Clone()
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = $args[$i]
    switch ($a) {
        { $_ -in @('--preset') }          { $PRESET = $args[++$i] }
        { $_ -in @('--num-gpus') }        { $NUM_GPUS = [int]$args[++$i] }
        { $_ -in @('--hours') }           { $HOURS = [int]$args[++$i] }
        { $_ -in @('--data') }            { $DATA_FILES += $args[++$i] }
        { $_ -in @('--output') }          { $OUTPUT_DIR = $args[++$i] }
        { $_ -in @('--seq-len') }         { $SEQ_LEN = [int]$args[++$i] }
        { $_ -in @('--micro-batch') }     { $MICRO_BATCH = [int]$args[++$i] }
        { $_ -in @('--grad-acc') }        { $GRAD_ACC = [int]$args[++$i] }
        { $_ -in @('--fp8') }             { $FP8 = "true" }
        { $_ -in @('--no-fp8') }          { $FP8 = "false" }
        { $_ -in @('--cpu-offload') }     { $CPU_OFFLOAD = "true" }
        { $_ -in @('--no-cpu-offload') }  { $CPU_OFFLOAD = "false" }
        { $_ -in @('--resume') }          { $RESUME_CKPT = $args[++$i] }
        { $_ -in @('--dry-run') }         { $DRY_RUN = "true" }
        { $_ -in @('-h','--help') } {
            @"
Usage: powershell -File scripts\run_turbo.ps1 [options]

Training preset:
  --preset NAME          12B_dense | 30B_dense | 60B_moe | 100B_moe | 600B_moe (required)

Resources:
  --num-gpus N           Number of GPUs (default: 8)
  --hours H              Max training hours (default: 24)

Data & output:
  --data PATH            Training JSONL (repeatable, required)
  --output PATH          Output directory (default: ./output)

Training hyperparameters:
  --seq-len N            Sequence length (default: 2048)
  --micro-batch N        Micro batch size per GPU (default: 4)
  --grad-acc N           Gradient accumulation steps (default: 8)

Optimizations:
  --fp8 | --no-fp8       FP8 mixed precision (default: on)
  --cpu-offload          ZeRO CPU offload (default: off)

Resume & debug:
  --resume PATH          Resume from checkpoint directory
  --dry-run              Run only steps 0-2, do not launch training
"@
            exit 0
        }
        default { Write-Fatal "Unknown argument: $a" }
    }
}

# ---------- Validate ----------
if ([string]::IsNullOrWhiteSpace($PRESET)) { Write-Fatal "--preset is required" }
if ($DATA_FILES.Count -eq 0) { Write-Fatal "--data is required (at least one JSONL)" }
foreach ($f in $DATA_FILES) {
    if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { Write-Fatal "Data file not found: $f" }
}
if (-not [string]::IsNullOrWhiteSpace($RESUME_CKPT)) {
    if (-not (Test-Path -LiteralPath $RESUME_CKPT -PathType Container)) { Write-Fatal "Resume checkpoint dir not found: $RESUME_CKPT" }
}

# ---------- Directories & Logs ----------
$null = New-Item -ItemType Directory -Force -Path (Join-Path $OUTPUT_DIR "logs")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $OUTPUT_DIR "configs")
$null = New-Item -ItemType Directory -Force -Path (Join-Path $OUTPUT_DIR "checkpoints")
$TIMESTAMP = (Get-Date -Format "yyyyMMdd_HHmmss")
$LOG_FILE = Join-Path $OUTPUT_DIR "logs/run_${TIMESTAMP}.log"
$PID_FILE = Join-Path $OUTPUT_DIR "run_${TIMESTAMP}.pid"
$DS_CONFIG = Join-Path $OUTPUT_DIR "configs/ds_config_${TIMESTAMP}.json"

# Tee to file + host
try {
    Start-Transcript -Path $LOG_FILE -UseMinimalHeader -ErrorAction SilentlyContinue | Out-Null
    $usingTranscript = $true
} catch {
    $usingTranscript = $false
    Write-Warn "Start-Transcript unavailable; falling back to redirection-only logging"
}

$PID | Out-File -FilePath $PID_FILE -Encoding utf8

# ========================================================================
# Step 0: Environment Check
# ========================================================================
Step-Start 0 "环境检查 (Python / CUDA / GPU 数量)"

# --- Python ---
try {
    $PY = Get-Command python -ErrorAction Stop
    Write-Info ("Python command: " + $PY.Source)
} catch {
    try {
        $PY = Get-Command py -ErrorAction Stop
        Write-Info ("Python command: " + $PY.Source + " (use py launcher)")
    } catch {
        Write-Fatal "Python not found in PATH"
    }
}
function Invoke-Python([string[]]$pyArgs) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = "python"
    foreach ($a in $pyArgs) { $psi.ArgumentList.Add($a) | Out-Null }
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $p = [System.Diagnostics.Process]::Start($psi)
    $out = $p.StandardOutput.ReadToEnd()
    $err = $p.StandardError.ReadToEnd()
    $p.WaitForExit()
    [pscustomobject]@{ ExitCode = $p.ExitCode; StdOut = $out; StdErr = $err }
}

$pyVer = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
Write-Info ("Python version: " + ($pyVer ?? "N/A"))

# --- GPU / nvidia-smi ---
try {
    $smi = Get-Command nvidia-smi -ErrorAction Stop
    $visible = if ($env:CUDA_VISIBLE_DEVICES) { $env:CUDA_VISIBLE_DEVICES } else { "all" }
    Write-Info ("CUDA_VISIBLE_DEVICES: $visible")
    $gpuList = (& nvidia-smi --query-gpu=name --format=csv,noheader) 2>$null
    $physical = if ($gpuList) { $gpuList.Count } else { 0 }
    Write-Info ("Physical GPUs detected:   $physical")
    Write-Info ("Requested num-gpus:       $NUM_GPUS")
    & nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>$null | Select-Object -First ([int]$NUM_GPUS + 1) | Out-Host
} catch {
    Write-Warn "nvidia-smi not found - GPU training may not work"
    $physical = 0
}

# --- CUDA toolkit ---
try {
    $nvccVer = (& nvcc --version 2>$null) -join "`n"
    if ($nvccVer -match 'release\s+(\d+\.\d+)') { Write-Info ("CUDA toolchain: release " + $matches[1]) }
} catch {
    Write-Warn "nvcc not found - proceeding anyway"
}

# --- Python sanity check ---
$pyCheck = @'
import sys
try:
    import torch
    has_torch = True
    ver = torch.__version__
    cuda = torch.version.cuda
    devc = torch.cuda.device_count() if torch.cuda.is_available() else 0
    avail = torch.cuda.is_available()
except Exception:
    has_torch, ver, cuda, devc, avail = False, "N/A", "N/A", 0, False
try:
    import deepspeed; ds = deepspeed.__version__
except Exception:
    ds = "N/A"
try:
    import transformer_engine; te = transformer_engine.__version__
except Exception:
    te = "N/A"
print(f"  torch            = {ver}")
print(f"  torch.cuda       = {cuda}")
print(f"  cuda.is_available= {avail}")
print(f"  cuda.device_count= {devc}")
print(f"  deepspeed        = {ds}")
print(f"  transformer_engine = {te}")
'@
$pyCheck | Out-File -FilePath (Join-Path ([System.IO.Path]::GetTempPath()) "_qxt_check.py") -Encoding utf8
$r = Invoke-Python @((Join-Path ([System.IO.Path]::GetTempPath()) "_qxt_check.py"))
if ($r.ExitCode -ne 0) { Write-Fatal "Python / PyTorch sanity check failed"; exit 1 }
Write-Info $r.StdOut

# --- Detect launcher mode ---
try {
    $null = Get-Command deepspeed -ErrorAction Stop
    $HAS_DEEPSPEED = $true
} catch { $HAS_DEEPSPEED = $false }
try {
    $null = Get-Command torchrun -ErrorAction Stop
    $HAS_TORCHRUN = $true
} catch { $HAS_TORCHRUN = $false }

if ($HAS_DEEPSPEED) {
    Write-Info "deepspeed launcher detected - will attempt to use it"
    if ($IsWindows) {
        Write-Warn "Windows detected - 'deepspeed' command has limited native support, falling back to torchrun/DDP"
        $LAUNCH_MODE = "torchrun"
    } else {
        $LAUNCH_MODE = "deepspeed"
    }
} elseif ($HAS_TORCHRUN) {
    Write-Info "torchrun detected - will use DDP mode"
    $LAUNCH_MODE = "torchrun"
} else {
    Write-Info "No distributed launcher detected - will use plain python -m (single process)"
    $LAUNCH_MODE = "python"
}

Step-Done 0

# ========================================================================
# Step 1: Budget Analysis
# ========================================================================
Step-Start 1 "训练预算分析 (budget_scheduler --dry-run)"

$budgetArgs = @("-m","qxt.training.budget_scheduler",
    "--dry-run",
    "--preset", $PRESET,
    "--num-gpus", $NUM_GPUS,
    "--hours", $HOURS,
    "--seq-len", $SEQ_LEN,
    "--micro-batch", $MICRO_BATCH,
    "--grad-acc", $GRAD_ACC,
    "--fp8", $FP8,
    "--cpu-offload", $CPU_OFFLOAD,
    "--output", $OUTPUT_DIR)
foreach ($f in $DATA_FILES) { $budgetArgs += @("--data", $f) }

$r = & python @budgetArgs
$ec = $LASTEXITCODE
if ($ec -ne 0) { Write-Fatal "budget_scheduler failed" $ec }
Write-Info "budget_scheduler --dry-run completed"

Step-Done 1

# ========================================================================
# Step 2: DeepSpeed Config
# ========================================================================
Step-Start 2 "生成 DeepSpeed 配置 (deepspeed_config.py)"

$dsArgs = @("-m","qxt.training.deepspeed_config",
    "--preset", $PRESET,
    "--num-gpus", $NUM_GPUS,
    "--fp8", $FP8,
    "--cpu-offload", $CPU_OFFLOAD,
    "--output", $DS_CONFIG)
$r = & python @dsArgs
$ec = $LASTEXITCODE
if ($ec -ne 0) { Write-Fatal "deepspeed_config generation failed" $ec }
Write-Info ("DeepSpeed config written to: $DS_CONFIG")

Step-Done 2

# ========================================================================
# Step 3: Launch Training
# ========================================================================
if ($DRY_RUN -eq "true") {
    Write-Warn "--dry-run: skipping Step 3 (training launch) and Step 4 (report)"
    Write-Warn "Review budget analysis + DS config, then rerun without --dry-run."
    Write-Host ""
    Write-Host "  _______________________________"  -ForegroundColor Green
    Write-Host " /     DRY-RUN  COMPLETED        \" -ForegroundColor Green
    Write-Host " \  ready to train when you are  /" -ForegroundColor Green
    Write-Host "  -------------------------------"  -ForegroundColor Green
    Write-Host "         \   ^__^"                    -ForegroundColor Green
    Write-Host "          \  (oo)\_______"            -ForegroundColor Green
    Write-Host "             (__)\       )\/\ /\"     -ForegroundColor Green
    Write-Host "                 ||----w |"           -ForegroundColor Green
    Write-Host "                 ||     ||"           -ForegroundColor Green
    Write-Host ""
    if ($usingTranscript) { Stop-Transcript | Out-Null }
    exit 0
}

Step-Start 3 "启动分布式训练 (turbo_distiller.py via $LAUNCH_MODE)"

$commonArgs = @("--preset", $PRESET,
    "--seq-len", $SEQ_LEN,
    "--micro-batch", $MICRO_BATCH,
    "--grad-acc", $GRAD_ACC,
    "--fp8", $FP8,
    "--cpu-offload", $CPU_OFFLOAD,
    "--hours", $HOURS,
    "--deepspeed-config", $DS_CONFIG,
    "--output", $OUTPUT_DIR)
if (-not [string]::IsNullOrWhiteSpace($RESUME_CKPT)) { $commonArgs += @("--resume", $RESUME_CKPT) }
foreach ($f in $DATA_FILES) { $commonArgs += @("--data", $f) }

# Build launcher
switch ($LAUNCH_MODE) {
    "deepspeed" {
        $launcherArgs = @("--num_gpus=$NUM_GPUS", "-m", "qxt.training.turbo_distiller")
        Write-Info "Launch mode: deepspeed --num_gpus=$NUM_GPUS -m qxt.training.turbo_distiller"
        & deepspeed @launcherArgs @commonArgs
        $ec = $LASTEXITCODE
    }
    "torchrun" {
        $launcherArgs = @("--nproc_per_node=$NUM_GPUS", "-m", "qxt.training.turbo_distiller")
        Write-Info "Launch mode: torchrun --nproc_per_node=$NUM_GPUS -m qxt.training.turbo_distiller (DDP)"
        & torchrun @launcherArgs @commonArgs
        $ec = $LASTEXITCODE
    }
    default {
        Write-Warn "No distributed launcher found. Running single-process python -m (training may be very slow)"
        $launcherArgs = @("-m", "qxt.training.turbo_distiller")
        & python @launcherArgs @commonArgs
        $ec = $LASTEXITCODE
    }
}

if ($ec -ne 0) { Write-Fatal "Training process exited with non-zero status" $ec }

Step-Done 3

# ========================================================================
# Step 4: Summary Report
# ========================================================================
Step-Start 4 "训练结束，自动汇总报告 (monitor.py)"

try {
    $r = & python -m qxt.training.monitor --mode summary --output $OUTPUT_DIR
    $ec = $LASTEXITCODE
    if ($ec -ne 0) { Write-Warn "monitor.py summary step failed - check logs manually" }
    else { Write-Info "Summary report generated" }
} catch {
    Write-Warn "monitor.py summary step failed - check logs manually"
}

Write-Info ("PID file:        $PID_FILE")
Write-Info ("Log file:        $LOG_FILE")
Write-Info ("Config file:     $DS_CONFIG")
Write-Info ("Checkpoints:     $(Join-Path $OUTPUT_DIR 'checkpoints')")
Write-Info ("TensorBoard:     $(Join-Path $OUTPUT_DIR 'tb')")

Write-Host ""
$banner = @"
  ___________________________________________________________
 /                                                           \
|   QINGXIAOTUAN LLM TRAINING COMPLETED SUCCESSFULLY         |
|                                                           |
|   preset      : ${PRESET}
|   num-gpus    : ${NUM_GPUS}
|   seq-len     : ${SEQ_LEN}
|   micro-batch : ${MICRO_BATCH}
|   grad-acc    : ${GRAD_ACC}
|   fp8         : ${FP8}
|   cpu-offload : ${CPU_OFFLOAD}
|   output      : ${OUTPUT_DIR}
|   log         : ${LOG_FILE}
 \___________________________________________________________/
      \   ^__^
       \  (oo)\_______
          (__)\       )\/\ /
              ||----w |
              ||     ||
"@
Write-Color -text $banner -color Green

if ($usingTranscript) {
    try { Stop-Transcript | Out-Null } catch {}
}

Step-Done 4
