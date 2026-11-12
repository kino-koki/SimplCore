param(
    [Parameter(Mandatory=$true)][string]$Teacher,
    [Parameter(Mandatory=$true)][string]$Tokenizer,
    [Parameter(Mandatory=$true)][string]$Data,
    [Parameter(Mandatory=$true)][string]$Out,
    [int]$Steps = 120,
    [int]$L = 24,
    [switch]$Bf16
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = 'python'

Write-Host "== 1/3 真迁移 (COPY/DERIVE/NEW) ==" -ForegroundColor Cyan
& $py (Join-Path $here 'migrate_qwen3.py') --teacher $Teacher --out $Out --tokenizer $Tokenizer
if ($LASTEXITCODE -ne 0) { throw "migrate failed ($LASTEXITCODE)" }

Write-Host "== 2/3 真实语料适配训练 ==" -ForegroundColor Cyan
$trainArgs = @((Join-Path $here 'train_simplcore.py'), '--model', $Out,
               '--data', $Data, '--tokenizer', $Tokenizer, '--out', $Out,
               '--steps', "$Steps", '--L', "$L")
if ($Bf16) { $trainArgs += '--bf16' }
& $py @trainArgs
if ($LASTEXITCODE -ne 0) { throw "train failed ($LASTEXITCODE)" }

Write-Host "== 3/3 硬验收 (delta_sgc / 流式 / 生成) ==" -ForegroundColor Cyan
$verifyArgs = @((Join-Path $here 'verify_arch.py'), '--model', $Out, '--tokenizer', $Tokenizer)
$rebuilt = Join-Path $Out '_rebuilt_teacher.safetensors'
if (Test-Path $rebuilt) {
    $verifyArgs += @('--teacher-file', $rebuilt)
} else {
    $verifyArgs += @('--teacher', $Teacher)
}
& $py @verifyArgs
if ($LASTEXITCODE -ne 0) { throw "verify failed ($LASTEXITCODE)" }

Write-Host "`nALL PASS. 结果: $(Join-Path $Out 'verify_arch.json')" -ForegroundColor Green
