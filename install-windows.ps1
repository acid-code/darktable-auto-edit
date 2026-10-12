#Requires -Version 5.1
<#
.SYNOPSIS
  Install the Darktable AI editor for the current Windows user.

.DESCRIPTION
  Creates a virtual environment, links this folder into Darktable's Lua
  directory, adds the luarc line, and points the AI editor Python preference
  at that virtual environment. Safe to run again.

  Run install-windows.cmd from this folder. Do not use "Run as administrator":
  an elevated window can see a different AppData than Darktable.

.PARAMETER Python
  Full path to python.exe. Skips the search.

.PARAMETER Darktable
  Full path to darktable.exe. Used for --print-paths and the version check.

.PARAMETER ConfigDir
  Darktable config directory. Defaults to the path darktable.exe reports,
  then %LOCALAPPDATA%\darktable.

.PARAMETER WithAi
  Also install requirements-ai.txt (YOLO, MediaPipe, Ollama client).

.PARAMETER InstallPython
  If no usable Python is found, install Python 3.12 with winget for this user.

.PARAMETER Copy
  Copy the project into the Lua folder instead of making a junction.

.PARAMETER Force
  Replace an existing junction that points somewhere else, re-enable a
  commented-out luarc line, and overwrite a custom Python preference.

.PARAMETER AllowOldDarktable
  Continue when the installed Darktable is older than 5.0.

.PARAMETER SkipSmoke
  Do not run the analyzer on a tiny generated image after install.
#>
[CmdletBinding()]
param(
    [string]$Python,
    [string]$Darktable,
    [string]$ConfigDir,
    [switch]$WithAi,
    [switch]$InstallPython,
    [switch]$Copy,
    [switch]$Force,
    [switch]$AllowOldDarktable,
    [switch]$SkipSmoke
)

Set-StrictMode -Version Latest

$script:PrefKey = 'lua/ai_editor/python'
$script:RequireLine = 'require "ai_editor/ai_editor"'
$script:LogPath = $null

function Write-Install {
    param(
        [Parameter(Mandatory = $true)][string]$Level,
        [Parameter(Mandatory = $true)][string]$Message
    )
    $line = "[{0}] {1}" -f $Level, $Message
    Write-Host $line
    if ($script:LogPath) {
        $utf8 = New-Object System.Text.UTF8Encoding $false
        $stamp = Get-Date -Format 'yyyy-MM-ddTHH:mm:ss'
        [System.IO.File]::AppendAllText($script:LogPath, ("{0} {1}{2}" -f $stamp, $line, [Environment]::NewLine), $utf8)
    }
}

function Get-NormalizedPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    $full = [System.IO.Path]::GetFullPath($Path)
    return $full.TrimEnd('\', '/')
}

function Test-SamePath {
    param([string]$Left, [string]$Right)
    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) {
        return $false
    }
    $a = (Get-NormalizedPath $Left).Replace('/', '\')
    $b = (Get-NormalizedPath $Right).Replace('/', '\')
    return [string]::Equals($a, $b, [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-UncPath {
    param([string]$Path)
    return $Path -match '^(//|\\\\)'
}

function Get-PrintedConfigDir {
    param([string]$Text)
    foreach ($line in ($Text -split "`r?`n")) {
        # darktable prints "configdir   C:\Users\...\darktable" with spaces, not a colon.
        if ($line -match '^\s*configdir\s+(.+?)\s*$') {
            return $Matches[1].Trim()
        }
    }
    return $null
}

function Test-StoreAliasPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    return $Path -match '(?i)[\\/]Microsoft[\\/]WindowsApps[\\/]'
}

function Test-SupportedPythonVersion {
    param([int]$Major, [int]$Minor)
    return ($Major -eq 3 -and $Minor -ge 9 -and $Minor -le 14)
}

function ConvertFrom-PythonProbe {
    param([string]$Text)
    $lines = @($Text -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($lines.Count -lt 5) {
        return $null
    }
    $major = 0
    $minor = 0
    $patch = 0
    $bits = 0
    if (-not [int]::TryParse($lines[0].Trim(), [ref]$major)) { return $null }
    if (-not [int]::TryParse($lines[1].Trim(), [ref]$minor)) { return $null }
    if (-not [int]::TryParse($lines[2].Trim(), [ref]$patch)) { return $null }
    if (-not [int]::TryParse($lines[3].Trim(), [ref]$bits)) { return $null }
    $executable = $lines[4].Trim()
    if ([string]::IsNullOrWhiteSpace($executable)) {
        return $null
    }
    return [pscustomobject]@{
        Major = $major
        Minor = $minor
        Patch = $patch
        Bits  = $bits
        Path  = $executable
    }
}

function Select-PythonCandidate {
    param($Candidates)
    $usable = @($Candidates | Where-Object {
            $_.Bits -eq 64 -and (Test-SupportedPythonVersion $_.Major $_.Minor)
        })
    if ($usable.Count -eq 0) {
        return $null
    }
    $preferred = @($usable | Where-Object { $_.Minor -le 13 })
    if ($preferred.Count -gt 0) {
        $usable = $preferred
    }
    return $usable |
        Sort-Object @{ Expression = { $_.Major }; Descending = $true },
                    @{ Expression = { $_.Minor }; Descending = $true },
                    @{ Expression = { $_.Patch }; Descending = $true } |
        Select-Object -First 1
}

function Read-TextPreserve {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [pscustomobject]@{ Text = ''; Bom = 'none'; Exists = $false }
    }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    $utf8 = New-Object System.Text.UTF8Encoding $false
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        return [pscustomobject]@{
            Text   = $utf8.GetString($bytes, 3, $bytes.Length - 3)
            Bom    = 'utf8'
            Exists = $true
        }
    }
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) {
        $text = [System.Text.Encoding]::Unicode.GetString($bytes, 2, $bytes.Length - 2)
        return [pscustomobject]@{ Text = $text; Bom = 'utf16-le'; Exists = $true }
    }
    if ($bytes.Length -ge 2 -and $bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) {
        $text = [System.Text.Encoding]::BigEndianUnicode.GetString($bytes, 2, $bytes.Length - 2)
        return [pscustomobject]@{ Text = $text; Bom = 'utf16-be'; Exists = $true }
    }
    return [pscustomobject]@{ Text = $utf8.GetString($bytes); Bom = 'none'; Exists = $true }
}

function Write-TextPreserve {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [Parameter(Mandatory = $true)][string]$Bom
    )
    $encoding = New-Object System.Text.UTF8Encoding ($Bom -eq 'utf8')
    $parent = Split-Path -Parent $Path
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Get-NewlineStyle {
    param([string]$Text)
    if ($Text.Contains("`r`n")) {
        return "`r`n"
    }
    if ($Text.Contains("`n")) {
        return "`n"
    }
    return "`r`n"
}

function Split-KeptLines {
    param([string]$Text)
    if ($null -eq $Text -or $Text -eq '') {
        return
    }
    # [regex]::Split keeps the empty field after a trailing newline.
    # The -split operator with a negative limit returns the file as one line.
    foreach ($part in [regex]::Split($Text, "\r\n|\n")) {
        $part
    }
}

function New-LineList {
    param([string]$Text)
    $list = New-Object 'System.Collections.Generic.List[string]'
    foreach ($part in (Split-KeptLines $Text)) {
        $list.Add([string]$part)
    }
    Write-Output -NoEnumerate $list
}

function Test-ActiveRequireLine {
    param([string]$Line)
    $code = $Line
    $comment = $code.IndexOf('--')
    if ($comment -ge 0) {
        $code = $code.Substring(0, $comment)
    }
    return $code -match 'require\s*\(?\s*[''"]ai_editor/ai_editor[''"]\s*\)?'
}

function Test-CommentedRequireLine {
    param([string]$Line)
    if (Test-ActiveRequireLine $Line) {
        return $false
    }
    return $Line -match 'require\s*\(?\s*[''"]ai_editor/ai_editor[''"]\s*\)?'
}

function Update-Luarc {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [switch]$EnableCommented
    )
    $read = Read-TextPreserve $Path
    if ($read.Bom -like 'utf16*') {
        Write-Install 'WARN' "luarc was UTF-16 and would not load in Darktable. Rewriting it as UTF-8."
    }
    $lineEnding = Get-NewlineStyle $read.Text
    $lines = New-LineList $read.Text
    $active = $false
    $commented = $false
    foreach ($line in $lines) {
        if (Test-ActiveRequireLine $line) { $active = $true }
        if (Test-CommentedRequireLine $line) { $commented = $true }
    }
    if ($active) {
        return 'present'
    }
    if ($commented -and -not $EnableCommented) {
        return 'commented'
    }
    if (-not $read.Exists -or $read.Text.Trim().Length -eq 0) {
        $lines = New-Object 'System.Collections.Generic.List[string]'
        $lines.Add('-- Load the AI editor at Darktable startup.')
    }
    else {
        $lines.Add('-- Added by the AI editor installer.')
    }
    $lines.Add($script:RequireLine)
    $text = [string]::Join($lineEnding, $lines.ToArray())
    if (-not $text.EndsWith($lineEnding)) {
        $text += $lineEnding
    }
    $bom = $read.Bom
    if ($bom -like 'utf16*') { $bom = 'none' }
    Write-TextPreserve -Path $Path -Text $text -Bom $bom
    if (-not $read.Exists) { return 'created' }
    return 'added'
}

function Update-ConfValue {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value
    )
    if ($Value -match "[\r\n]") {
        throw "The Python path contains a newline, so it cannot be stored in darktablerc."
    }
    $limit = 1000
    if (($Key.Length + 1 + $Value.Length) -gt $limit) {
        throw "The Python path is too long for darktablerc (Darktable reads at most about 1000 characters per line)."
    }
    $read = Read-TextPreserve $Path
    if ($read.Bom -like 'utf16*') {
        Write-Install 'WARN' "darktablerc was UTF-16. Rewriting it as UTF-8 so Darktable keeps every setting."
    }
    $lineEnding = Get-NewlineStyle $read.Text
    $lines = New-LineList $read.Text
    $match = New-Object System.Collections.Generic.List[int]
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if ($line -match '^\s*#') { continue }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { continue }
        $found = $line.Substring(0, $eq)
        if ($found -eq $Key) {
            $match.Add($i)
        }
    }
    $newLine = '{0}={1}' -f $Key, $Value
    $changed = $false
    if ($match.Count -eq 0) {
        if ($lines.Count -gt 0 -and $lines[$lines.Count - 1] -eq '') {
            $lines.Insert($lines.Count - 1, $newLine)
        }
        else {
            $lines.Add($newLine)
        }
        $changed = $true
    }
    elseif ($lines[$match[0]] -ne $newLine -or $match.Count -gt 1) {
        $lines[$match[0]] = $newLine
        for ($i = $match.Count - 1; $i -ge 1; $i--) {
            $lines.RemoveAt($match[$i])
        }
        $changed = $true
    }
    if (-not $changed) {
        return $false
    }
    $text = [string]::Join($lineEnding, $lines.ToArray())
    if (-not $text.EndsWith("`n")) {
        $text += $lineEnding
    }
    $bom = $read.Bom
    if ($bom -like 'utf16*') { $bom = 'none' }
    $backup = '{0}.ai-editor.bak' -f $Path
    if ($read.Exists -and -not (Test-Path -LiteralPath $backup)) {
        Copy-Item -LiteralPath $Path -Destination $backup
        Write-Install 'OK' "Backed up darktablerc to $backup"
    }
    Write-TextPreserve -Path $Path -Text $text -Bom $bom
    return $true
}

function Get-ConfValue {
    param([string]$Path, [string]$Key)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $read = Read-TextPreserve $Path
    foreach ($line in (New-LineList $read.Text)) {
        if ($line -match '^\s*#') { continue }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { continue }
        if ($line.Substring(0, $eq) -eq $Key) {
            return $line.Substring($eq + 1)
        }
    }
    return $null
}

function Test-ManagedPythonCommand {
    param([string]$Current, [string]$VenvPython)
    if ([string]::IsNullOrWhiteSpace($Current)) { return $true }
    $trimmed = $Current.Trim().Trim('"').Trim("'")
    $bare = @('python', 'python3', 'py', 'python.exe', 'python3.exe', 'py.exe')
    if ($bare -contains $trimmed.ToLowerInvariant()) { return $true }
    try {
        if (Test-SamePath $trimmed $VenvPython) { return $true }
        $venvRoot = Split-Path -Parent (Split-Path -Parent $VenvPython)
        $full = (Get-NormalizedPath $trimmed).Replace('/', '\')
        $root = (Get-NormalizedPath $venvRoot).Replace('/', '\')
        if ($full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    catch {
        return $false
    }
    return $false
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [int]$TimeoutSec = 25
    )
    $start = New-Object System.Diagnostics.ProcessStartInfo
    $start.FileName = $File
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.CreateNoWindow = $true
    $start.Arguments = ($ArgumentList | ForEach-Object {
            if ($_ -match '[\s"]') { '"{0}"' -f ($_ -replace '"', '\"') } else { $_ }
        }) -join ' '
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $start
    [void]$proc.Start()
    $stdoutTask = $proc.StandardOutput.ReadToEndAsync()
    $stderrTask = $proc.StandardError.ReadToEndAsync()
    if (-not $proc.WaitForExit($TimeoutSec * 1000)) {
        try { $proc.Kill() } catch { }
        throw "Timed out after $TimeoutSec seconds: $File"
    }
    $proc.WaitForExit()
    return [pscustomobject]@{
        Code = $proc.ExitCode
        Out  = $stdoutTask.Result
        Err  = $stderrTask.Result
    }
}

function Get-ReparseTarget {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $item = Get-Item -LiteralPath $Path -Force
    $isLink = [bool]($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)
    if (-not $isLink) { return $null }
    if ($item.PSObject.Properties.Name -contains 'Target' -and $item.Target) {
        $target = $item.Target
        if ($target -is [array]) { return [string]$target[0] }
        return [string]$target
    }
    $parent = Split-Path -Parent $Path
    $name = Split-Path -Leaf $Path
    $listing = & cmd.exe /c "dir /AL `"$parent`""
    foreach ($line in @($listing)) {
        if ($line -match '<(JUNCTION|SYMLINKD|SYMLINK)>\s+(.*?)\s+\[(.+)\]\s*$') {
            if ($Matches[2].Trim() -eq $name) {
                return $Matches[3].Trim()
            }
        }
    }
    return ''
}

function Remove-DirectoryLink {
    param([Parameter(Mandatory = $true)][string]$Path)
    # rmdir removes a junction or symlink. Remove-Item -Recurse can delete the target.
    & cmd.exe /c "rmdir `"$Path`""
    if ($LASTEXITCODE -ne 0) {
        throw "Could not remove the existing link at $Path (exit $LASTEXITCODE)."
    }
}

function Test-DirectoryWritable {
    param([Parameter(Mandatory = $true)][string]$Dir)
    $probeDir = $Dir
    if (-not (Test-Path -LiteralPath $probeDir)) {
        $probeDir = Split-Path -Parent $probeDir
    }
    if (-not $probeDir -or -not (Test-Path -LiteralPath $probeDir)) {
        return $false
    }
    $probe = Join-Path $probeDir ('.ai-editor-write-' + [guid]::NewGuid().ToString('n'))
    try {
        $utf8 = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllText($probe, 'ok', $utf8)
        Remove-Item -LiteralPath $probe -Force
        return $true
    }
    catch {
        return $false
    }
}

function Test-ProjectLayout {
    param([string]$Root)
    $required = @(
        'ai_editor.lua',
        'ai_processor.py',
        'requirements.txt',
        'lib\json.lua',
        'ai_dt\pipeline.py',
        'recipes\general\faithful.json',
        'guides\general.json'
    )
    $missing = @()
    foreach ($rel in $required) {
        $path = Join-Path $Root ($rel -replace '\\', [IO.Path]::DirectorySeparatorChar)
        if (-not (Test-Path -LiteralPath $path)) { $missing += $rel }
    }
    return $missing
}

function Main {
    $ErrorActionPreference = 'Stop'
    if (-not $env:OS -or $env:OS -ne 'Windows_NT') {
        Write-Host '[FAIL] This installer is for Windows. On Linux or macOS, follow the steps in README.md.'
        return 2
    }

    $project = $PSScriptRoot
    if ([string]::IsNullOrWhiteSpace($project)) {
        Write-Host '[FAIL] Could not find the folder this script lives in.'
        return 2
    }
    $missing = @(Test-ProjectLayout $project)
    if ($missing.Count -gt 0) {
        Write-Host ("[FAIL] This script must stay next to the AI editor files. Missing: {0}" -f ($missing -join ', '))
        return 2
    }

    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
    if ($principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host '[WARN] This window is running as Administrator.'
        Write-Host '[WARN] Darktable uses the normal account AppData. Run install-windows.cmd without elevating.'
    }

    $running = @(Get-Process -Name 'darktable', 'darktable-cli' -ErrorAction SilentlyContinue)
    if ($running.Count -gt 0) {
        Write-Host '[FAIL] Quit Darktable completely, then run the installer again.'
        Write-Host '[FAIL] Darktable rewrites darktablerc when it exits, which would drop the Python path set while it is open.'
        return 1
    }

    $config = Resolve-ConfigDir -DarktableExe $Darktable -Requested $ConfigDir
    if (-not (Test-Path -LiteralPath $config.Path)) {
        New-Item -ItemType Directory -Path $config.Path -Force | Out-Null
        Write-Install 'OK' "Created config directory $($config.Path)"
    }
    $script:LogPath = Join-Path $config.Path 'ai-editor-install.log'
    Write-Install 'OK' "Project: $project"
    Write-Install 'OK' "Darktable config: $($config.Path) ($($config.Source))"
    if ($config.Version) {
        Write-Install 'OK' "Darktable version: $($config.Version)"
        $major = 0
        if ($config.Version -match '^(\d+)\.') { $major = [int]$Matches[1] }
        if ($major -gt 0 -and $major -lt 5 -and -not $AllowOldDarktable) {
            Write-Install 'FAIL' 'Darktable 5.0 or newer is required. Pass -AllowOldDarktable to continue anyway.'
            return 1
        }
        if ($major -gt 0 -and $major -lt 5) {
            Write-Install 'WARN' 'This Darktable is older than 5.0. The style it imports may not apply.'
        }
    }
    else {
        Write-Install 'WARN' 'Darktable.exe was not found, so the version was not checked. Install Darktable 5 or newer, or pass -Darktable.'
    }

    $extraRc = @(Get-ChildItem -LiteralPath $config.Path -Filter 'darktablerc-*' -File -ErrorAction SilentlyContinue)
    if ($extraRc.Count -gt 0) {
        Write-Install 'WARN' 'Other workspace files (darktablerc-*) were left alone. This install updates the default darktablerc only.'
    }

    $pythonExe = Resolve-PythonExe -Requested $Python -AllowInstall:$InstallPython
    Write-Install 'OK' ("Python {0}.{1}.{2} ({3}-bit): {4}" -f $pythonExe.Major, $pythonExe.Minor, $pythonExe.Patch, $pythonExe.Bits, $pythonExe.Path)
    if ($pythonExe.Minor -ge 14) {
        Write-Install 'WARN' 'Python 3.14 or newer may not have NumPy wheels yet. If pip fails, install Python 3.12 and pass -Python.'
    }

    $venvPython = Install-VirtualEnv -BasePython $pythonExe.Path -Project $project -Config $config.Path -Recreate:$Force
    Write-Install 'OK' "Virtual environment: $venvPython"
    Install-Requirements -VenvPython $venvPython -Project $project -WithAi:$WithAi
    if (-not $SkipSmoke) {
        Invoke-SmokeTest -VenvPython $venvPython -Project $project
    }

    $luaDir = Join-Path $config.Path 'lua'
    if (-not (Test-Path -LiteralPath $luaDir)) {
        New-Item -ItemType Directory -Path $luaDir -Force | Out-Null
    }
    $dest = Join-Path $luaDir 'ai_editor'
    $linked = Install-PluginTree -Source $project -Destination $dest -Copy:$Copy -ForceLink:$Force
    Write-Install 'OK' $linked

    $luarc = Join-Path $config.Path 'luarc'
    $luarcResult = Update-Luarc -Path $luarc -EnableCommented:$Force
    switch ($luarcResult) {
        'created' { Write-Install 'OK' "Created $luarc" }
        'added' { Write-Install 'OK' "Added the AI editor line to $luarc" }
        'present' { Write-Install 'OK' 'luarc already loads the AI editor.' }
        'commented' {
            Write-Install 'WARN' 'luarc has the AI editor commented out, so it was left disabled. Pass -Force to enable it again.'
        }
    }

    $rcPath = Join-Path $config.Path 'darktablerc'
    $current = Get-ConfValue -Path $rcPath -Key $script:PrefKey
    $stored = $venvPython
    if ($current -and -not (Test-ManagedPythonCommand -Current $current -VenvPython $venvPython) -and -not $Force) {
        Write-Install 'WARN' "Leaving the existing Python preference: $current"
        Write-Install 'WARN' 'Pass -Force to point AI editor: Python command at this virtual environment.'
        $stored = $current
    }
    else {
        $wrote = Update-ConfValue -Path $rcPath -Key $script:PrefKey -Value $venvPython
        if ($wrote) {
            Write-Install 'OK' "Set $script:PrefKey"
        }
        else {
            Write-Install 'OK' 'Python preference already points at this virtual environment.'
        }
    }

    Write-Install 'OK' 'Install finished.'
    Write-Host ''
    Write-Host 'Start Darktable. On the lighttable, select a photo and press Run AI Analysis.'
    Write-Host 'If the panel is hidden, right-click the right panel and enable AI Editor.'
    Write-Host 'Do not also enable ai_editor from the scripts module, or the panel is registered twice.'
    Write-Host 'Leave lib\json.lua disabled if it is listed. It is a library, not a second plugin.'
    if ($stored) {
        Write-Host "Python command: $stored"
    }
    return 0
}

function Resolve-ConfigDir {
    param([string]$DarktableExe, [string]$Requested)
    if ($Requested) {
        return [pscustomobject]@{
            Path    = (Get-NormalizedPath $Requested)
            Source  = 'the -ConfigDir argument'
            Version = $(if ($DarktableExe) { Get-DarktableVersion $DarktableExe } else { $null })
        }
    }
    $exe = Find-DarktableExe -Requested $DarktableExe
    $version = $null
    $fromPrint = $null
    if ($exe) {
        $version = Get-DarktableVersion $exe
        try {
            $printed = Invoke-Captured -File $exe -ArgumentList @('--print-paths') -TimeoutSec 20
            if ($printed.Code -eq 0) {
                $fromPrint = Get-PrintedConfigDir $printed.Out
            }
        }
        catch {
            Write-Install 'WARN' "darktable --print-paths failed: $($_.Exception.Message)"
        }
    }
    if ($fromPrint) {
        return [pscustomobject]@{ Path = (Get-NormalizedPath $fromPrint); Source = 'darktable --print-paths'; Version = $version }
    }
    $local = $env:LOCALAPPDATA
    if ([string]::IsNullOrWhiteSpace($local)) {
        $local = [Environment]::GetFolderPath('LocalApplicationData')
    }
    if ([string]::IsNullOrWhiteSpace($local)) {
        throw 'LOCALAPPDATA is empty, so the Darktable config directory cannot be chosen. Pass -ConfigDir.'
    }
    return [pscustomobject]@{
        Path    = (Join-Path $local 'darktable')
        Source  = '%LOCALAPPDATA%\darktable'
        Version = $version
    }
}

function Get-DarktableVersion {
    param([string]$Exe)
    try {
        $result = Invoke-Captured -File $Exe -ArgumentList @('--version') -TimeoutSec 20
        $blob = ($result.Out + "`n" + $result.Err)
        if ($blob -match 'darktable\s+(\d+\.\d+(?:\.\d+)?)') {
            return $Matches[1]
        }
    }
    catch {
        return $null
    }
    return $null
}

function Get-NoteProperty {
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

function Find-DarktableExe {
    param([string]$Requested)
    if ($Requested) {
        if (-not (Test-Path -LiteralPath $Requested)) {
            throw "Darktable was not found at $Requested"
        }
        return (Get-NormalizedPath $Requested)
    }
    $candidates = New-Object System.Collections.Generic.List[string]
    $cmd = Get-Command 'darktable.exe' -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { $candidates.Add($cmd.Source) }
    $roots = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKCU:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )
    foreach ($root in $roots) {
        # Some uninstall keys have no DisplayName. Strict mode throws on a missing note property.
        $apps = @(Get-ItemProperty $root -ErrorAction SilentlyContinue | Where-Object {
                $name = Get-NoteProperty $_ 'DisplayName'
                $name -and ([string]$name -match 'darktable')
            })
        foreach ($app in $apps) {
            $location = Get-NoteProperty $app 'InstallLocation'
            if ($location) {
                $candidates.Add((Join-Path $location 'bin\darktable.exe'))
                $candidates.Add((Join-Path $location 'darktable.exe'))
            }
            $displayIcon = Get-NoteProperty $app 'DisplayIcon'
            if ($displayIcon) {
                $icon = ([string]$displayIcon) -replace ',.*$', ''
                $icon = $icon.Trim('"')
                if ($icon) { $candidates.Add($icon) }
            }
        }
    }
    $programFiles = @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)
    foreach ($base in $programFiles) {
        if ($base) {
            $candidates.Add((Join-Path $base 'darktable\bin\darktable.exe'))
            $candidates.Add((Join-Path $base 'Programs\darktable\bin\darktable.exe'))
        }
    }
    if ($env:USERPROFILE) {
        $candidates.Add((Join-Path $env:USERPROFILE 'scoop\apps\darktable\current\bin\darktable.exe'))
        $candidates.Add((Join-Path $env:USERPROFILE 'scoop\apps\darktable\current\darktable.exe'))
    }
    $shortcutRoots = @(
        (Join-Path $env:ProgramData 'Microsoft\Windows\Start Menu\Programs'),
        (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
    )
    $shell = $null
    foreach ($root in $shortcutRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $links = @(Get-ChildItem -LiteralPath $root -Filter '*darktable*.lnk' -Recurse -ErrorAction SilentlyContinue)
        foreach ($link in $links) {
            if (-not $shell) { $shell = New-Object -ComObject WScript.Shell }
            $target = $shell.CreateShortcut($link.FullName).TargetPath
            if ($target) { $candidates.Add($target) }
        }
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return (Get-NormalizedPath $candidate)
        }
    }
    return $null
}

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = (@($user, $machine) | Where-Object { $_ }) -join ';'
}

function Resolve-PythonExe {
    param([string]$Requested, [switch]$AllowInstall)
    if ($Requested) {
        $probed = Probe-Python $Requested
        if (-not $probed) { throw "Could not run $Requested" }
        if ($probed.Bits -ne 64) { throw "$Requested is 32-bit. NumPy needs 64-bit Python." }
        if ($probed.Major -lt 3 -or $probed.Minor -lt 9) { throw "$Requested is too old. Install Python 3.9 or newer." }
        return $probed
    }
    $found = @(Find-PythonCandidates)
    $best = Select-PythonCandidate $found
    if ($best) { return $best }
    if ($AllowInstall) {
        Install-Python312
        $found = @(Find-PythonCandidates)
        $best = Select-PythonCandidate $found
        if ($best) { return $best }
    }
    throw @"
No 64-bit Python 3.9 through 3.13 was found.
The Microsoft Store alias (WindowsApps\python.exe) does not count.
Install Python from https://www.python.org/downloads/windows/ and enable the py launcher,
or re-run with -InstallPython to let winget install Python 3.12 for this user.
"@
}

function Install-Python312 {
    $winget = Get-Command 'winget.exe' -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw 'winget is not available, so Python cannot be installed automatically.'
    }
    Write-Install 'OK' 'Installing Python 3.12 for the current user with winget.'
    & winget.exe install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "winget could not install Python 3.12 (exit $LASTEXITCODE)."
    }
    Update-SessionPath
}

function Find-PythonCandidates {
    $paths = New-Object System.Collections.Generic.List[string]
    $py = Get-Command 'py.exe' -ErrorAction SilentlyContinue
    if ($py -and -not (Test-StoreAliasPath $py.Source)) {
        try {
            $listed = Invoke-Captured -File $py.Source -ArgumentList @('-0p') -TimeoutSec 20
            foreach ($line in ($listed.Out -split "`r?`n")) {
                if ($line -match '([A-Za-z]:\\.*python\.exe)\s*$') {
                    $paths.Add($Matches[1])
                }
            }
        }
        catch {
            Write-Install 'WARN' "py -0p failed: $($_.Exception.Message)"
        }
    }
    $scanRoots = @()
    if ($env:LOCALAPPDATA) { $scanRoots += (Join-Path $env:LOCALAPPDATA 'Programs\Python') }
    if ($env:ProgramFiles) { $scanRoots += (Join-Path $env:ProgramFiles 'Python') }
    $scanRoots += 'C:\Python312', 'C:\Python311', 'C:\Python310', 'C:\Python39'
    if ($env:USERPROFILE) {
        $scanRoots += (Join-Path $env:USERPROFILE 'miniconda3')
        $scanRoots += (Join-Path $env:USERPROFILE 'anaconda3')
        $scanRoots += (Join-Path $env:USERPROFILE 'scoop\apps\python\current')
    }
    foreach ($root in $scanRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $direct = Join-Path $root 'python.exe'
        if (Test-Path -LiteralPath $direct) { $paths.Add($direct) }
        $children = @(Get-ChildItem -LiteralPath $root -Filter 'python.exe' -Recurse -Depth 2 -ErrorAction SilentlyContinue)
        foreach ($child in $children) { $paths.Add($child.FullName) }
    }
    $where = Get-Command 'python.exe' -ErrorAction SilentlyContinue
    if ($where -and $where.Source) { $paths.Add($where.Source) }

    $seen = @{}
    $result = @()
    foreach ($path in $paths) {
        if (-not $path -or (Test-StoreAliasPath $path)) { continue }
        $key = $path.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $probed = Probe-Python $path
        if ($probed) { $result += $probed }
    }
    return $result
}

function Probe-Python {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    if (Test-StoreAliasPath $Path) { return $null }
    $item = Get-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
    if ($item -and $item.Length -eq 0) { return $null }
    $probe = Join-Path ([System.IO.Path]::GetTempPath()) 'ai-editor-pyprobe.py'
    $code = @"
import struct, sys
print(sys.version_info[0])
print(sys.version_info[1])
print(sys.version_info[2])
print(struct.calcsize("P") * 8)
print(sys.executable)
"@
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($probe, $code, $utf8)
    try {
        $result = Invoke-Captured -File $Path -ArgumentList @($probe) -TimeoutSec 20
    }
    catch {
        return $null
    }
    if ($result.Code -ne 0) { return $null }
    return ConvertFrom-PythonProbe $result.Out
}

function Install-VirtualEnv {
    param(
        [string]$BasePython,
        [string]$Project,
        [string]$Config,
        [switch]$Recreate
    )
    $primary = Join-Path $Project '.venv'
    if (-not (Test-DirectoryWritable $Project)) {
        $fallback = Join-Path $Config 'ai-editor-venv'
        Write-Install 'WARN' "The project folder is not writable, so the virtual environment will be $fallback"
        return Initialize-Venv -BasePython $BasePython -VenvRoot $fallback -Recreate:$Recreate
    }
    try {
        return Initialize-Venv -BasePython $BasePython -VenvRoot $primary -Recreate:$Recreate
    }
    catch {
        $message = $_.Exception.Message
        if ($message -match '(?i)access|denied|permission|read-only|not writable') {
            $fallback = Join-Path $Config 'ai-editor-venv'
            Write-Install 'WARN' "Cannot create .venv in the project folder ($message)."
            Write-Install 'WARN' "Using $fallback instead."
            return Initialize-Venv -BasePython $BasePython -VenvRoot $fallback -Recreate:$Recreate
        }
        throw
    }
}

function Initialize-Venv {
    param([string]$BasePython, [string]$VenvRoot, [switch]$Recreate)
    $venvPython = Join-Path $VenvRoot 'Scripts\python.exe'
    if ($Recreate -and (Test-Path -LiteralPath $VenvRoot)) {
        Write-Install 'OK' "Removing $VenvRoot"
        & cmd.exe /c "rmdir /s /q `"$VenvRoot`""
        if (Test-Path -LiteralPath $VenvRoot) {
            throw "Could not remove $VenvRoot. Close any Python window using it and delete that folder."
        }
    }
    if (Test-Path -LiteralPath $venvPython) {
        $check = Probe-Python $venvPython
        if ($check) { return (Get-NormalizedPath $venvPython) }
        Write-Install 'WARN' 'The existing virtual environment does not run. Recreating it.'
        & cmd.exe /c "rmdir /s /q `"$VenvRoot`""
    }
    Write-Install 'OK' "Creating a virtual environment with $BasePython"
    $venvOutput = & $BasePython -m venv $VenvRoot 2>&1
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython)) {
        $detail = ($venvOutput | Out-String).Trim()
        if ($detail) { $detail = " $detail" }
        throw "Could not create a virtual environment at $VenvRoot.$detail Install Python from python.org with the py launcher and pip options enabled. The embeddable zip does not include venv."
    }
    return (Get-NormalizedPath $venvPython)
}

function Install-Requirements {
    param([string]$VenvPython, [string]$Project, [switch]$WithAi)
    Write-Install 'OK' 'Installing NumPy and Pillow.'
    & $VenvPython -m pip install --disable-pip-version-check --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        Write-Install 'WARN' 'pip could not upgrade itself. Continuing with the pip that came with the virtual environment.'
    }
    $req = Join-Path $Project 'requirements.txt'
    & $VenvPython -m pip install --disable-pip-version-check -r $req
    if ($LASTEXITCODE -ne 0) {
        throw "pip could not install requirements.txt (exit $LASTEXITCODE). If this machine uses a proxy, set HTTPS_PROXY and run the installer again."
    }
    if ($WithAi) {
        $ai = Join-Path $Project 'requirements-ai.txt'
        Write-Install 'OK' 'Installing the optional YOLO, MediaPipe, and Ollama packages.'
        & $VenvPython -m pip install --disable-pip-version-check -r $ai
        if ($LASTEXITCODE -ne 0) {
            throw "pip could not install requirements-ai.txt (exit $LASTEXITCODE)."
        }
    }
}

function Invoke-SmokeTest {
    param([string]$VenvPython, [string]$Project)
    $dir = Join-Path ([System.IO.Path]::GetTempPath()) ('ai-editor-smoke-' + [guid]::NewGuid().ToString('n'))
    New-Item -ItemType Directory -Path $dir | Out-Null
    $png = Join-Path $dir 'sample.png'
    $maker = @"
from PIL import Image
Image.new('RGB', (48, 36), (24, 24, 24)).save(r'''$png''')
"@
    $makerPath = Join-Path $dir 'make.py'
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($makerPath, $maker, $utf8)
    & $VenvPython $makerPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $png)) {
        throw 'The virtual environment cannot import Pillow. The requirements install did not finish cleanly.'
    }
    $processor = Join-Path $Project 'ai_processor.py'
    & $VenvPython $processor $png --output-dir $dir --detector heuristic --advisor histogram
    if ($LASTEXITCODE -ne 0) {
        throw "The analyzer smoke test failed (exit $LASTEXITCODE)."
    }
    $json = Join-Path $dir 'output.json'
    $style = Join-Path $dir 'edit.dtstyle'
    if (-not (Test-Path -LiteralPath $json) -or -not (Test-Path -LiteralPath $style)) {
        throw 'The smoke test ran but did not write output.json and edit.dtstyle.'
    }
    Write-Install 'OK' 'Analyzer smoke test passed.'
    Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction SilentlyContinue
}

function Install-PluginTree {
    param([string]$Source, [string]$Destination, [switch]$Copy, [switch]$ForceLink)
    if (Test-SamePath $Source $Destination) {
        return "The project is already the Darktable copy at $Destination"
    }
    $exists = Test-Path -LiteralPath $Destination
    if ($exists) {
        $target = Get-ReparseTarget $Destination
        if ($null -eq $target) {
            throw @"
$Destination already exists as a real folder.
The installer will not delete it.
Quit Darktable, rename that folder, and run the installer again.
"@
        }
        if ($target -eq '') {
            if (-not $ForceLink) {
                throw "$Destination is a link whose target could not be read. Pass -Force to replace the link."
            }
        }
        elseif (Test-SamePath $target $Source) {
            return "Darktable already uses this project via $Destination"
        }
        elseif (-not $ForceLink) {
            throw "$Destination already points at $target. Pass -Force to retarget it at this project."
        }
        Remove-DirectoryLink $Destination
    }
    if (-not $Copy -and -not (Test-UncPath $Source)) {
        try {
            New-Item -ItemType Junction -Path $Destination -Target $Source | Out-Null
            return "Linked $Destination -> $Source (junction, so edits in this folder are what Darktable runs)"
        }
        catch {
            Write-Install 'WARN' "Could not create a junction ($($_.Exception.Message)). Copying the files instead."
        }
    }
    if (Test-UncPath $Source) {
        Write-Install 'WARN' 'The project is on a network path, so it will be copied. Run the installer again after you change it.'
    }
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    & robocopy.exe $Source $Destination /E /NFL /NDL /NJH /NJS /XD .git .venv __pycache__ .pytest_cache ai-editor-work agent-tools .mypy_cache
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed with exit $LASTEXITCODE while copying into $Destination."
    }
    return "Copied the project to $Destination. Run this installer again after pulling updates."
}

if ($MyInvocation.InvocationName -ne '.') {
    try {
        $code = Main
    }
    catch {
        Write-Host ("[FAIL] {0}" -f $_.Exception.Message)
        $code = 1
    }
    exit $code
}
