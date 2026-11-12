<#
.SYNOPSIS
  用 gh-cli 登录态，从私有仓 kino-koki/Qwen3-1.7B-Instruct 拉全量权重到本地。

.USAGE
  powershell -ExecutionPolicy Bypass -File .\pull_qwen3_gh.ps1
  powershell -ExecutionPolicy Bypass -File .\pull_qwen3_gh.ps1 -Dst 'G:\Gpan\model'
  powershell -ExecutionPolicy Bypass -File .\pull_qwen3_gh.ps1 -Dst 'G:\model' -Throttle 6 -Reassemble

.NOTES
  * 先在 GH 上登录一次：gh auth login --web --scopes repo
    或直接给环境变量：$env:GH_TOKEN = '<PAT, 需私有仓 Contents: read>'
  * 千万不要用 PowerShell 的 ">" 重定向接收 gh api 的二进制输出——
    Windows PowerShell 5.1 的 ">" 是文本(UTF-16)重定向，会把 99 MB 权重写坏。
    本脚本用 Invoke-WebRequest -OutFile，逐字节落盘。
  * 全部 55 文件 / 3.789 GiB；断点续传：已存在且字节数正确即跳过。
#>
param(
    [string]$Dst = 'G:\model\Qwen3-1.7B-Instruct',
    [int]$Throttle = 4,
    [switch]$Reassemble
)

$ErrorActionPreference = 'Stop'
$Repo         = 'kino-koki/Qwen3-1.7B-Instruct'
$ManifestRepo = 'kino-koki/SimplCore'
$ManifestPath = 'weights/Qwen3-1.7B-Instruct_manifest.json'

# ---------- 0. gh 登录态 ----------
$tok = $env:GH_TOKEN
if (-not $tok) {
    gh auth status | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '未登录，启动 gh auth login ...' -ForegroundColor Yellow
        gh auth login --web --scopes 'repo'
    }
    $tok = (gh auth token).Trim()
}
if (-not $tok) { throw '拿不到 token：先 gh auth login，或设置 $env:GH_TOKEN' }
Write-Host "gh token 已就绪 (长度 $($tok.Length))" -ForegroundColor Green

New-Item -ItemType Directory -Force -Path $Dst | Out-Null
Write-Host "目标目录: $Dst" -ForegroundColor Cyan

# ---------- 1. 权威文件清单（Git tree API，含精确 size）----------
$treeRaw = gh api "repos/$Repo/git/trees/main?recursive=1" | Out-String
$tree = $treeRaw | ConvertFrom-Json
$files = @($tree.tree | Where-Object { $_.type -eq 'blob' -and $_.size -gt 0 } | Sort-Object path)
$total = ($files | Measure-Object -Property size -Sum).Sum
Write-Host ("清单: {0} 个文件, 合计 {1:N3} GiB" -f $files.Count, ($total / 1GB))

# ---------- 2. 下载 ----------
$H = @{
    Authorization = "Bearer $tok"
    Accept        = 'application/vnd.github.raw'   # 真实 LFS 字节，不是 133B 指针
}
$UriBase = "https://api.github.com/repos/$Repo/contents"

$pull = {
    param($f, $dstDir, $hdr, $ub)
    $out = Join-Path $dstDir $f.path
    $dir = Split-Path -Parent $out
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    if (Test-Path $out) {
        if ((Get-Item $out).Length -eq $f.size) { return "SKIP  $($f.path)" }
    }
    $url = "$ub/$($f.path)?ref=main"
    $tmp = "$out.part"
    for ($a = 1; $a -le 4; $a++) {
        try {
            Invoke-WebRequest -Uri $url -Headers $hdr -UserAgent 'gh-qwen3-pull' `
                              -OutFile $tmp -UseBasicParsing -TimeoutSec 900
            break
        } catch {
            if ($a -eq 4) { throw }
            Start-Sleep -Seconds (2 * $a)
        }
    }
    $n = (Get-Item $tmp).Length
    if ($n -ne $f.size) { Remove-Item $tmp -Force; throw "SIZE MISMATCH $($f.path): $n != $($f.size)" }
    Move-Item $tmp $out -Force
    return "OK    $($f.path)  $n B"
}

$t0 = Get-Date
if ($PSVersionTable.PSVersion.Major -ge 7 -and $Throttle -gt 1) {
    Write-Host "PowerShell 7 并行模式, Throttle=$Throttle" -ForegroundColor Cyan
    $files | ForEach-Object -Parallel {
        & $using:pull $_ $using:Dst $using:H $using:UriBase
    } -ThrottleLimit $Throttle | ForEach-Object { Write-Host $_ }
} else {
    Write-Host '顺序模式（装 PowerShell 7 可加 -Throttle 6 提速）' -ForegroundColor Cyan
    foreach ($f in $files) { Write-Host (& $pull $f $Dst $H $UriBase) }
}
Write-Host ("下载耗时 {0:N1} 分钟" -f ((Get-Date) - $t0).TotalMinutes)

# ---------- 3. size 复核 ----------
$badSize = @($files | Where-Object {
    $p = Join-Path $Dst $_.path
    (-not (Test-Path $p)) -or ((Get-Item $p).Length -ne $_.size)
})
Write-Host ("SIZE 复核: {0}/{1} 通过" -f ($files.Count - $badSize.Count), $files.Count)
if ($badSize.Count) { $badSize | ForEach-Object { Write-Warning "异常: $($_.path)" } }

# ---------- 4. sha256 复核（清单在公开仓，gh api 直取）----------
$mfRaw = gh api "repos/$ManifestRepo/contents/$ManifestPath?ref=main" `
                 -H 'Accept: application/vnd.github.raw' | Out-String
$mf = $mfRaw | ConvertFrom-Json
$bad = 0
foreach ($e in $mf.files) {
    $p = Join-Path $Dst $e.path
    if (-not (Test-Path $p)) { Write-Warning "MISSING  $($e.path)"; $bad++; continue }
    $h = (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($h -ne $e.sha256.ToLowerInvariant()) { Write-Warning "SHA256 MISMATCH  $($e.path)"; $bad++ }
}
Write-Host ("SHA256 复核: {0}/{1} 通过  (manifest 共 {2} 条)" -f ($mf.files.Count - $bad), $mf.files.Count, $mf.files.Count) `
           -ForegroundColor $(if ($bad -eq 0) { 'Green' } else { 'Red' })

# ---------- 5. 可选：qa_* / qb_* 拼回原始字节流 ----------
if ($Reassemble) {
    foreach ($pre in @('qa', 'qb')) {
        $pattern = $pre + '_*'
        $parts = @(Get-ChildItem -LiteralPath $Dst | Where-Object { $_.Name -like $pattern } | Sort-Object Name)
        $out = Join-Path $Dst ($pre + '_concat.bin')
        $fs = [System.IO.File]::Create($out)
        foreach ($p in $parts) {
            $b = [System.IO.File]::ReadAllBytes($p.FullName)
            $fs.Write($b, 0, $b.Length)
        }
        $fs.Dispose()
        Write-Host ("{0}: {1} 片 -> {2} ({3:N1} MiB)" -f $pre, $parts.Count, $out, ((Get-Item $out).Length / 1MB))
    }
}

if ($bad -eq 0 -and $badSize.Count -eq 0) {
    Write-Host "`nDONE — 全量权重已落盘且逐文件字节校验通过。" -ForegroundColor Green
} else {
    Write-Host "`n有异常，重跑本脚本即可断点续传（不会重复下载已完整的文件）。" -ForegroundColor Yellow
}
