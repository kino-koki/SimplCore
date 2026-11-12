param(
    [string]$Teacher = '',
    [string]$Tokenizer = '',
    [string]$Data = '',
    [string]$Out = '',
    [int]$Steps = 160
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = 'python'

Write-Host "== 1/3 全架构白盒验收 (ARCH_STEP 8) ==" -ForegroundColor Cyan
& $py (Join-Path $here 'verify_final.py') --steps $Steps --seed 23
if ($LASTEXITCODE -ne 0) { throw "verify_final failed ($LASTEXITCODE)" }

Write-Host "== 2/3 1.7B 生产链路 state_dict 契约 ==" -ForegroundColor Cyan
& $py (Join-Path $here 'contract_check.py')
if ($LASTEXITCODE -ne 0) { throw "contract_check failed ($LASTEXITCODE)" }

Write-Host "== 3/3 O(1) 状态生成演示 ==" -ForegroundColor Cyan
& $py (Join-Path $here 'stream_chat.py') "Q: Who goes there?`nA:" --n 120 --seed 7
if ($LASTEXITCODE -ne 0) { throw "stream_chat failed ($LASTEXITCODE)" }

if ($Teacher -and $Tokenizer -and $Data -and $Out) {
    Write-Host "== 可选：1.7B 真迁移 → 真实语料训练 → 硬验收 ==" -ForegroundColor Cyan
    & (Join-Path $here 'run_all.ps1') -Teacher $Teacher -Tokenizer $Tokenizer `
        -Data $Data -Out $Out -Steps 120
    if ($LASTEXITCODE -ne 0) { throw "run_all failed ($LASTEXITCODE)" }
} else {
    Write-Host "（未提供 -Teacher/-Tokenizer/-Data/-Out，跳过 1.7B 链路）" -ForegroundColor Yellow
}

Write-Host "`nALL PASS." -ForegroundColor Green
