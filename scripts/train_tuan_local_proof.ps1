<#
.SYNOPSIS
    在本机（无 CUDA 的 CPU / 核显主机）上跑一次 青小团 / Tuan 的真实语料收敛验证。

.DESCRIPTION
    目的不是刷分，而是证明「这条链路真的能训练」：真实中文语料、真实分词器、
    真实文档级切分、真实梯度更新、真实验证困惑度下降。

    诚实前提：本机（Ryzen 7 7735HS + 13.75 GB 内存 + CPU 版 torch）能承受的规模
    只够做**收敛性验证**，不足以支撑任何"强模型/SOTA"主张。更大的规模必须在
    1×A100 80GB 上跑，见 docs/SGC_A100_RUNBOOK.md。

.EXAMPLE
    pwsh scripts/train_tuan_local_proof.ps1
    pwsh scripts/train_tuan_local_proof.ps1 -Minutes 45 -SeqLen 512 -Dim 320 -Layers 6
#>
param(
    [string]$DataDir   = 'data\processed\tuan_zh',
    [string]$OutDir    = 'runs\local_proof',
    [int]   $Minutes   = 30,
    [int]   $SeqLen    = 512,
    [int]   $Dim       = 256,
    [int]   $Layers    = 4,
    [int]   $NumBases  = 32,
    [int]   $MicroBatch = 8,
    [int]   $MaxSteps  = 100000,
    [string]$Python    = '.venv311\Scripts\python.exe'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "0. 检查环境"
if (-not (Test-Path $Python)) { throw "找不到解释器 $Python" }
& $Python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "torch 不可用" }

Step "1. 检查语料"
if (-not (Test-Path (Join-Path $DataDir 'index.json'))) {
    Write-Host "未找到打包语料。请先运行：" -ForegroundColor Yellow
    Write-Host "  $Python scripts\download_corpus.py --out data\raw\fineweb_zh"
    Write-Host "  $Python scripts\build_corpus.py --parquet-dir data\raw\fineweb_zh --out $DataDir --vocab-size 32000 --seq-len 4096"
    throw "语料缺失：$DataDir\index.json"
}
$manifest = Get-Content (Join-Path $DataDir 'index.json') -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host ("vocab_size={0}  seq_len={1}  total_tokens={2:N0}" -f $manifest.vocab_size, $manifest.seq_len, $manifest.total_tokens)
foreach ($sp in 'train','val','test') {
    $e = $manifest.splits.$sp
    if ($e) { Write-Host ("  {0,-5} n_docs={1,6}  n_tokens={2,12:N0}  n_windows={3,8:N0}" -f $sp, $e.n_docs, $e.n_tokens, $e.n_windows) }
}
if (-not $manifest.splits.val) { Write-Host "警告: 没有 val 划分，将没有过拟合预警" -ForegroundColor Yellow }

Step "2. 冒烟：构建模型 + 前向/反向"
& $Python -c @"
import sys; sys.path.insert(0, r'$((Get-Location).Path)')
import torch
from simplcore.core.config import SimplCoreConfig
from simplcore.core.model import SimplCoreModel
cfg = SimplCoreConfig(vocab_size=$($manifest.vocab_size), dim=64, num_layers=2, num_bases=8)
m = SimplCoreModel(cfg); x = torch.randint(0, cfg.vocab_size, (2, 32))
_, loss = m(x, x); loss.backward()
print('smoke OK  loss=%.3f  params=%d' % (float(loss), m.num_params()))
"@
if ($LASTEXITCODE -ne 0) { throw "冒烟失败" }

Step "3. 训练（时长上限 $Minutes 分钟）"
& $Python -u -m simplcore.core.train `
    --data-dir $DataDir --out-dir $OutDir --seq-len $SeqLen `
    --override "dim=$Dim" "num_layers=$Layers" "num_bases=$NumBases" "vocab_size=$($manifest.vocab_size)" `
    --micro-batch $MicroBatch --grad-accum 4 `
    --lr 6e-4 --lr-min 6e-5 --warmup 200 --max-steps $MaxSteps `
    --eval-every 200 --eval-batches 20 --save-every 500 --log-every 50 `
    --num-workers 0 --time-budget-min $Minutes --seed 1337
if ($LASTEXITCODE -ne 0) { throw "训练失败" }

Step "4. 结果"
if (Test-Path (Join-Path $OutDir 'summary.json')) {
    Get-Content (Join-Path $OutDir 'summary.json') -Raw -Encoding UTF8
}
Write-Host "`n指标曲线: $OutDir\metrics.jsonl" -ForegroundColor Green
Write-Host "检查点:   $OutDir\best.pt / last.pt" -ForegroundColor Green
Write-Host "`n注意：这是**收敛性验证**，不是性能主张。规模上限见 docs/SGC_SPEC.md 第 8 节。" -ForegroundColor Yellow
