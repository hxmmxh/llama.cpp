[CmdletBinding()]
param(
    [Parameter(Position = 0)][string]$Model,
    [ValidateSet('cli', 'server', 'devices')][string]$Mode = 'cli',
    [string]$DraftModel,
    [string]$Prompt,
    [string]$PromptFile,
    [string]$PrefillFile,
    [ValidateSet('lf', 'preserve')][string]$PrefillNewlines = 'lf',
    [switch]$SkipPrefill,
    [ValidateRange(1, 86400)][int]$PrefillTimeoutSec = 1800,
    [ValidateRange(512, 1048576)][int]$ContextSize = 8192,
    [ValidateRange(1, 1024)][int]$Threads = 12,
    [ValidateRange(-1, 1048576)][int]$MaxTokens = 512,
    [ValidateRange(1, 7)][int]$DraftMax = 3,
    [ValidateSet('f16', 'q8_0', 'q4_0')][string]$KVCache = 'f16',
    [string]$Device = 'Vulkan0',
    [Alias('Host')][ValidateNotNullOrEmpty()][string]$ListenHost = '127.0.0.1',
    [ValidateRange(1, 65535)][int]$Port = 8080,
    [string]$ModelAlias = 'spark-x2.6-1.7b',
    [ValidateRange(-1, 2147483647)][int]$CacheRamMiB = 8192,
    [switch]$NoCacheIdleSlots,
    [ValidateRange(0, 1048576)][int]$ContextCheckpoints = 32,
    [ValidateRange(1, 1048576)][int]$CheckpointMinStep = 8192,
    [uint32]$Seed = 0,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$exeName = if ($Mode -eq 'cli') { 'llama-cli.exe' } else { 'llama-server.exe' }
$exe = Join-Path $PSScriptRoot ('bin\' + $exeName)
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "File not found: $exe" }

function Resolve-ModelFile([string]$Path) {
    $Path = $Path.Trim().Trim('"')
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "File not found: $Path" }
    return (Resolve-Path -LiteralPath $Path).ProviderPath
}

function Invoke-PrefilledServer([string[]]$NativeArguments, [string]$Prefix) {
    # Check the port before starting so prefill cannot reach an existing service.
    $listener = [Net.Sockets.TcpListener]::new($serverListenAddress, $Port)
    $listener.ExclusiveAddressUse = $true
    try { $listener.Start() } finally { $listener.Stop() }

    $server = [Diagnostics.Process]::new()
    $started = $false
    try {
        $server.StartInfo.FileName = $exe
        $server.StartInfo.UseShellExecute = $false
        # ProcessStartInfo on Windows takes a command line, not an argument array.
        $server.StartInfo.Arguments = ($NativeArguments | ForEach-Object {
            '"' + ($_ -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"'
        }) -join ' '
        if (-not $server.Start()) { throw 'Unable to start the server.' }
        $started = $true

        $baseUri = $serverBaseUri
        Write-Host 'Waiting for the server before prefix prefill...'
        $timer = [Diagnostics.Stopwatch]::StartNew()
        $ready = $false
        while (-not $ready -and $timer.Elapsed.TotalSeconds -lt $PrefillTimeoutSec) {
            if ($server.HasExited) { throw "Server exited before prefill (code $($server.ExitCode))." }
            try {
                $health = Invoke-RestMethod -Uri "$baseUri/health" -TimeoutSec 2
                $ready = $health.status -eq 'ok'
            } catch {
                $ready = $false
            }
            if (-not $ready) { Start-Sleep -Milliseconds 250 }
        }
        if (-not $ready) { throw "Server readiness timed out after $PrefillTimeoutSec seconds." }
        if ($server.HasExited) { throw "Server exited before prefill (code $($server.ExitCode))." }

        # The input is already rendered. Token IDs avoid adding another template or BOS.
        $tokenizeBody = @{
            content = $Prefix
            add_special = $false
            parse_special = $true
        } | ConvertTo-Json -Depth 4 -Compress
        $tokenized = Invoke-RestMethod -Uri "$baseUri/tokenize" -Method Post `
            -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($tokenizeBody)) -TimeoutSec $PrefillTimeoutSec
        $prefixTokens = @($tokenized.tokens)
        if (-not $prefixTokens.Count) { throw 'Prefix tokenization returned no tokens.' }
        $body = @{
            id_slot = 0
            prompt = $prefixTokens
            n_predict = 0
            cache_prompt = $true
            stream = $false
        } | ConvertTo-Json -Depth 4 -Compress
        Write-Host "Prefilling prefix from $PrefillFile..."
        $response = Invoke-RestMethod -Uri "$baseUri/completion" -Method Post `
            -ContentType 'application/json; charset=utf-8' `
            -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec $PrefillTimeoutSec
        # This engine reports one sampled token for n_predict=0; it is not committed to KV.
        # Discard that response text and validate the requested budget and cached input instead.
        if (-not $response.timings -or $response.generation_settings.n_predict -ne 0 -or
            $response.tokens_predicted -notin @(0, 1)) {
            throw 'Invalid prefill response: expected a zero-budget completion.'
        }
        $count = $response.timings.prompt_n + $response.timings.cache_n
        if ($count -ne $prefixTokens.Count) { throw 'Prefix prefill token count mismatch.' }
        Write-Host "Prefill complete: $count tokens in slot 0; prompt_ms=$($response.timings.prompt_ms)."
        Write-Host "HTTP API ready: $baseUri/v1"
        while (-not $server.WaitForExit(500)) { }
        return $server.ExitCode
    } finally {
        if ($started -and -not $server.HasExited) { $server.Kill(); $server.WaitForExit() }
        $server.Dispose()
    }
}

# Use the measured Arc 140T shader path for this release.
$previousCoopmat = $env:GGML_VK_DISABLE_COOPMAT
$env:GGML_VK_DISABLE_COOPMAT = '1'
try {
    if ($PrefillFile -and $Mode -ne 'server') { throw 'PrefillFile is only supported in server mode.' }
    if ($SkipPrefill -and $Mode -ne 'server') { throw 'SkipPrefill is only supported in server mode.' }
    if ($SkipPrefill -and $PrefillFile) { throw 'Use either PrefillFile or SkipPrefill.' }
    if ($Mode -eq 'devices') {
        & $exe --list-devices
        exit $LASTEXITCODE
    }
    if ($Mode -eq 'server') {
        $serverListenAddress = [Net.IPAddress]::Parse($ListenHost)
        $localAddress = $serverListenAddress
        if ($serverListenAddress.Equals([Net.IPAddress]::Any)) {
            $localAddress = [Net.IPAddress]::Loopback
        } elseif ($serverListenAddress.Equals([Net.IPAddress]::IPv6Any)) {
            $localAddress = [Net.IPAddress]::IPv6Loopback
        }
        $serverBaseUri = [UriBuilder]::new('http', $localAddress.ToString(), $Port).Uri.AbsoluteUri.TrimEnd('/')
        if (-not $PSBoundParameters.ContainsKey('Model')) { $Model = 'D:\models\Spark-X2.6-1.7B-Q4_K_M-Q4_1.gguf' }
        if (-not $PSBoundParameters.ContainsKey('DraftModel')) { $DraftModel = 'D:\models\Spark-X2.6-1.7B-DSpark-Q8_0-backbone.gguf' }
        if (-not $PSBoundParameters.ContainsKey('ContextSize')) { $ContextSize = 131072 }
        if (-not $PSBoundParameters.ContainsKey('MaxTokens')) { $MaxTokens = 131072 }
        if (-not $PSBoundParameters.ContainsKey('KVCache')) { $KVCache = 'q8_0' }
        if (-not $SkipPrefill -and -not $PSBoundParameters.ContainsKey('PrefillFile')) {
            $PrefillFile = Join-Path $PSScriptRoot 'astudio_template\template.txt'
        }
    }
    if (-not $Model) { $Model = Read-Host 'GGUF model path' }
    $modelPath = Resolve-ModelFile $Model
    if ($Prompt -and $PromptFile) { throw 'Use either Prompt or PromptFile.' }
    if ($Mode -eq 'server' -and ($Prompt -or $PromptFile)) { throw 'Send server prompts through the HTTP API.' }
    if ($PrefillFile) {
        $PrefillFile = Resolve-ModelFile $PrefillFile
        $prefix = [IO.File]::ReadAllText($PrefillFile, [Text.UTF8Encoding]::new($false, $true))
        if ($PrefillNewlines -eq 'lf') { $prefix = $prefix.Replace("`r`n", "`n") }
        if ([string]::IsNullOrWhiteSpace($prefix)) { throw 'PrefillFile must contain a UTF-8 prompt prefix.' }
    }
    $nativeArgs = @('-m', $modelPath, '--device', $Device, '-ngl', '99',
        '-fa', 'on', '-c', "$ContextSize", '-t', "$Threads", '-tb', "$Threads",
        '-b', '512', '-ub', '128', '-ctk', $KVCache, '-ctv', $KVCache,
        '--cache-type-k-swa', 'f16', '--cache-type-v-swa', 'f16',
        '--fit', 'off', '--jinja', '--reasoning', 'on', '--reasoning-effort', 'low',
        '--temp', '1', '--top-p', '0.95', '--top-k', '0', '--min-p', '0',
        '--repeat-penalty', '1', '--presence-penalty', '0', '--frequency-penalty', '0',
        '-n', "$MaxTokens")
    if ($DraftModel) {
        $nativeArgs += @('--spec-type', 'draft-dspark', '-md', (Resolve-ModelFile $DraftModel),
            '--device-draft', $Device, '-ngld', '99', '-td', "$Threads", '-tbd', "$Threads",
            '-ctkd', 'f16', '-ctvd', 'f16', '--spec-draft-n-max', "$DraftMax",
            '--spec-draft-p-min', '0', '--spec-draft-sampling', 'probabilistic',
            '--no-spec-draft-backend-sampling')
    } else {
        $nativeArgs += @('--spec-type', 'none')
    }
    if ($Mode -eq 'server') {
        $nativeArgs += @('--host', $ListenHost, '--port', "$Port", '-np', '1',
            '--seed', "$Seed", '--kv-v-token32', '--cache-prompt',
            '--cache-ram', "$CacheRamMiB", '--ctx-checkpoints', "$ContextCheckpoints",
            '--checkpoint-min-step', "$CheckpointMinStep", '--no-webui')
        if ($ModelAlias) { $nativeArgs += @('--alias', $ModelAlias) }
        if ($NoCacheIdleSlots -or $CacheRamMiB -eq 0) {
            $nativeArgs += '--no-cache-idle-slots'
        } else {
            $nativeArgs += '--cache-idle-slots'
        }
        Write-Host "Listening address: $ListenHost; port: $Port"
        if (-not $PrefillFile) { Write-Host "HTTP API: $serverBaseUri/v1" }
    } elseif ($PromptFile) {
        $nativeArgs += @('--single-turn', '-f', (Resolve-ModelFile $PromptFile))
    } elseif ($PSBoundParameters.ContainsKey('Prompt')) {
        $nativeArgs += @('--single-turn', '-p', $Prompt)
    }
    $nativeArgs += $ExtraArgs
    if ($PrefillFile) {
        # Prefill targets slot 0 on this server; preserve the configured RAM cache.
        $nativeArgs += @('--host', $ListenHost, '--port', "$Port", '-np', '1',
            '--cache-prompt')
        $result = Invoke-PrefilledServer -NativeArguments $nativeArgs -Prefix $prefix
    } else {
        & $exe @nativeArgs
        $result = $LASTEXITCODE
    }
} finally {
    $env:GGML_VK_DISABLE_COOPMAT = $previousCoopmat
}
exit $result
