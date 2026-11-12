param(
    [string]$Repo = 'kino-koki/SimplCore',
    [string]$Branch = 'main',
    [string]$RemoteDir = 'arch_v8'
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$tmp = Join-Path $here '_push_clone'

$token = (gh auth token).Trim()
if (-not $token) { throw 'gh 未登录：先执行 gh auth login --web --scopes repo' }

if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
git clone --depth 1 "https://github.com/$Repo.git" $tmp
if ($LASTEXITCODE -ne 0) { throw "clone failed ($LASTEXITCODE)" }

$dst = Join-Path $tmp $RemoteDir
if (Test-Path $dst) { Remove-Item -Recurse -Force $dst }
Copy-Item -Recurse -Force $here $dst
Get-ChildItem -Recurse -Force $dst -Include '__pycache__','_push_clone' |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# local integrity before push
$local = @{}
Get-ChildItem -Recurse -File $dst | ForEach-Object {
    $rel = $_.FullName.Substring($dst.Length + 1).Replace('\','/')
    $bytes = [System.IO.File]::ReadAllBytes($_.FullName)
    $hdr = [System.Text.Encoding]::ASCII.GetBytes("blob $($bytes.Length)`0")
    $all = New-Object byte[] ($hdr.Length + $bytes.Length)
    [Array]::Copy($hdr, 0, $all, 0, $hdr.Length)
    [Array]::Copy($bytes, 0, $all, $hdr.Length, $bytes.Length)
    $sha1 = [System.Security.Cryptography.SHA1]::Create().ComputeHash($all)
    $local[$rel] = -join ($sha1 | ForEach-Object { $_.ToString('x2') })
}
Write-Host "本地文件 $($local.Count) 个，逐文件 blob sha1 已算" -ForegroundColor Cyan

Push-Location $tmp
try {
    git add $RemoteDir
    $n = (git diff --cached --name-only | Measure-Object -Line).Lines
    if ($n -eq 0) { Write-Host '无新增内容，跳过提交'; return }
    git -c user.name='SimplCore' -c user.email='simplcore@local' `
        commit -m 'ARCH_STEP 8: trained generative SimplCore + O(1) streaming + 1.7B production path'
    git -c "http.extraHeader=AUTHORIZATION: bearer $token" push origin "HEAD:$Branch"
    if ($LASTEXITCODE -ne 0) { throw "push failed ($LASTEXITCODE)" }

    $tree = (gh api "repos/$Repo/git/trees/$Branch`?recursive=1" | Out-String |
             ConvertFrom-Json).tree
    $map = @{}
    foreach ($e in $tree) { if ($e.type -eq 'blob') { $map[$e.path] = $e.sha } }
    $ok = 0; $bad = 0
    foreach ($k in $local.Keys) {
        $remote = "$RemoteDir/$k"
        if ($map.ContainsKey($remote) -and $map[$remote] -eq $local[$k]) { $ok++ }
        else { $bad++; Write-Host "  MISMATCH $remote" }
    }
    Write-Host "远端回读: $ok OK / $bad MISMATCH"
    if ($bad -gt 0) { throw '远端校验未通过' }
} finally {
    Pop-Location
}
Write-Host "推送完成: https://github.com/$Repo/tree/$Branch/$RemoteDir" -ForegroundColor Green
