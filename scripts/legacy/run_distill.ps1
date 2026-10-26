$ErrorActionPreference = 'Continue'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$instructionFile = Join-Path $scriptDir 'distill_instructions.json'
$outputDir = Join-Path $scriptDir 'distillation_data'

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$apiKey = '***REMOVED***'
$apiUrl = 'https://apihub.agnes-ai.com/v1/chat/completions'
$model = 'agnes-2.0-flash'

$jsonContent = [System.IO.File]::ReadAllText($instructionFile, [System.Text.Encoding]::UTF8)
$instructions = ConvertFrom-Json -InputObject $jsonContent
$total = $instructions.Count

Write-Host "========== Agnes AI Distillation ==========" -ForegroundColor Cyan
Write-Host "  Total: $total"
Write-Host "  Model: $model"
Write-Host ""

function CallAPI($msg) {
    $escaped = $msg -replace '\\', '\'
    $escaped = $escaped -replace '"', '\"'
    $escaped = $escaped -replace "`r", ''
    $escaped = $escaped -replace "`n", '\n'
    $jsonText = '{"model":"' + $model + '","messages":[{"role":"user","content":"' + $escaped + '"}]'
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($jsonText)

    $headers = @{}
    $headers['Authorization'] = 'Bearer ' + $apiKey
    $headers['Content-Type'] = 'application/json'

    for ($retry = 0; $retry -lt 3; $retry++) {
        try {
            $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Headers $headers -Body $bytes -TimeoutSec 120
            if ($response -and $response.choices -and $response.choices.Count -gt 0) {
                return $response.choices[0].message.content
            }
            return $null
        } catch {
            if ($retry -lt 2) {
                Start-Sleep -Seconds 2
                continue
            }
            return $null
        }
    }
    return $null
}

$success = 0
$failed = 0

for ($i = 0; $i -lt $total; $i++) {
    $inst = $instructions[$i]
    $num = ($i + 1).ToString().PadLeft(2, '0')
    $text = $inst.text
    $type = $inst.type

    Write-Host "  [$num/$total] [$type] " -NoNewline

    $answer = CallAPI $text

    if ($answer -and $answer.Length -gt 0) {
        $success++
        Write-Host "OK ($($answer.Length) chars)" -ForegroundColor Green

        $output = @{ instruction = $text; input = ''; output = $answer; model = $model; task_type = $type }
        $line = ConvertTo-Json -InputObject $output -Depth 5 -Compress
        $line = $line + "`n"

        $typeFile = Join-Path $outputDir ($type + '.jsonl')
        [System.IO.File]::AppendAllText($typeFile, $line, [System.Text.Encoding]::UTF8)

        $allFile = Join-Path $outputDir 'all_samples.jsonl'
        [System.IO.File]::AppendAllText($allFile, $line, [System.Text.Encoding]::UTF8)
    } else {
        $failed++
        Write-Host "FAILED" -ForegroundColor Red
    }

    Start-Sleep -Milliseconds 300
}

Write-Host ""
Write-Host "========== Done ==========" -ForegroundColor Cyan
Write-Host "  Success: $success / $total" -ForegroundColor Green
Write-Host "  Failed: $failed" -ForegroundColor Red
Write-Host "  Output: $outputDir"
Write-Host ""

$allFile = Join-Path $outputDir 'all_samples.jsonl'
if (Test-Path $allFile) {
    $count = ([System.IO.File]::ReadAllLines($allFile, [System.Text.Encoding]::UTF8)).Count
    Write-Host "  all_samples.jsonl: $count entries"
}
foreach ($t in @('general', 'code', 'math', 'creative', 'logic')) {
    $f = Join-Path $outputDir ($t + '.jsonl')
    if (Test-Path $f) {
        $c = ([System.IO.File]::ReadAllLines($f, [System.Text.Encoding]::UTF8)).Count
        Write-Host "  $t.jsonl: $c entries"
    }
}
