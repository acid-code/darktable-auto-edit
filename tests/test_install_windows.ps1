# Tests for the Windows installer that do not need Darktable or Windows.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
. (Join-Path $root 'install-windows.ps1')

$failures = New-Object System.Collections.Generic.List[string]

function Assert-Equal($Name, $Actual, $Expected) {
    if ($Actual -ne $Expected) {
        $script:failures.Add("${Name}: expected [$Expected] but got [$Actual]")
    }
}

function Assert-True($Name, $Condition) {
    if (-not $Condition) {
        $script:failures.Add("${Name}: expected true")
    }
}

Assert-True 'store alias' (Test-StoreAliasPath 'C:\Users\A\AppData\Local\Microsoft\WindowsApps\python.exe')
Assert-True 'real python is not a store alias' (-not (Test-StoreAliasPath 'C:\Users\A\AppData\Local\Programs\Python\Python312\python.exe'))
Assert-True '3.12 supported' (Test-SupportedPythonVersion 3 12)
Assert-True '3.8 rejected' (-not (Test-SupportedPythonVersion 3 8))
Assert-True '3.14 still parseable' (Test-SupportedPythonVersion 3 14)

$probe = ConvertFrom-PythonProbe "3`r`n12`r`n1`r`n64`r`nC:\Py\python.exe`r`n"
Assert-Equal 'probe minor' $probe.Minor 12
Assert-Equal 'probe bits' $probe.Bits 64
Assert-True 'short probe rejected' ($null -eq (ConvertFrom-PythonProbe "3`n12"))

$candidates = @(
    [pscustomobject]@{ Major = 3; Minor = 8; Patch = 0; Bits = 64; Path = 'old' },
    [pscustomobject]@{ Major = 3; Minor = 14; Patch = 0; Bits = 64; Path = 'new' },
    [pscustomobject]@{ Major = 3; Minor = 11; Patch = 2; Bits = 32; Path = 'x86' },
    [pscustomobject]@{ Major = 3; Minor = 12; Patch = 4; Bits = 64; Path = 'good' },
    [pscustomobject]@{ Major = 3; Minor = 11; Patch = 9; Bits = 64; Path = 'older' }
)
$chosen = Select-PythonCandidate $candidates
Assert-Equal 'prefers 3.12 over 3.14 and 32-bit' $chosen.Path 'good'
Assert-Equal 'only 3.14 left' (Select-PythonCandidate @([pscustomobject]@{ Major = 3; Minor = 14; Patch = 1; Bits = 64; Path = 'new' })).Path 'new'
Assert-True 'nothing usable' ($null -eq (Select-PythonCandidate @([pscustomobject]@{ Major = 3; Minor = 8; Patch = 0; Bits = 64; Path = 'old' })))

Assert-True 'active require' (Test-ActiveRequireLine 'require "ai_editor/ai_editor"')
Assert-True 'call require' (Test-ActiveRequireLine "require('ai_editor/ai_editor')")
Assert-True 'comment is not active' (-not (Test-ActiveRequireLine '-- require "ai_editor/ai_editor"'))
Assert-True 'comment is detected' (Test-CommentedRequireLine '-- require "ai_editor/ai_editor"')
Assert-True 'other script ignored' (-not (Test-ActiveRequireLine 'require "ai_editor/ai_editor_extra"'))

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('ai-editor-install-test-' + [guid]::NewGuid().ToString('n'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    $luarc = Join-Path $tmp 'luarc'
    Assert-Equal 'create luarc' (Update-Luarc $luarc) 'created'
    Assert-Equal 'second luarc' (Update-Luarc $luarc) 'present'
    $text = (Read-TextPreserve $luarc).Text
    Assert-True 'one require' (($text | Select-String -Pattern 'require "ai_editor/ai_editor"' -AllMatches).Matches.Count -eq 1)

    $commented = Join-Path $tmp 'luarc-off'
    $utf8 = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllText($commented, "-- require `"ai_editor/ai_editor`"`nrequire `"other`"`n", $utf8)
    Assert-Equal 'leave commented' (Update-Luarc $commented) 'commented'
    Assert-Equal 'force commented' (Update-Luarc $commented -EnableCommented) 'added'
    $forced = (Read-TextPreserve $commented).Text
    Assert-True 'comment kept' ($forced.Contains('-- require "ai_editor/ai_editor"'))
    Assert-True 'active added' ($forced.Contains("`nrequire `"ai_editor/ai_editor`""))

    $rc = Join-Path $tmp 'darktablerc'
    [System.IO.File]::WriteAllText($rc, "# keep me`r`nlua/other=1`r`n", $utf8)
    Assert-True 'add key' (Update-ConfValue $rc 'lua/ai_editor/python' 'C:\Py\python.exe')
    $rcText = (Read-TextPreserve $rc).Text
    Assert-True 'crlf kept' ($rcText.Contains("`r`n"))
    Assert-True 'comment kept in rc' ($rcText.StartsWith('# keep me'))
    Assert-Equal 'read back' (Get-ConfValue $rc 'lua/ai_editor/python') 'C:\Py\python.exe'
    Assert-True 'same value is not a change' (-not (Update-ConfValue $rc 'lua/ai_editor/python' 'C:\Py\python.exe'))

    $dup = Join-Path $tmp 'duprc'
    [System.IO.File]::WriteAllText($dup, "lua/ai_editor/python=old`nlua/other=1`nlua/ai_editor/python=older`n", $utf8)
    Assert-True 'collapse duplicates' (Update-ConfValue $dup 'lua/ai_editor/python' 'C:\New\python.exe')
    $dupText = (Read-TextPreserve $dup).Text
    Assert-Equal 'one python key' (@($dupText -split "`n" | Where-Object { $_ -like 'lua/ai_editor/python=*' })).Count 1
    Assert-True 'other key kept' ($dupText.Contains('lua/other=1'))

    $eq = Join-Path $tmp 'eqrc'
    [System.IO.File]::WriteAllText($eq, "keep=a=b`n", $utf8)
    Update-ConfValue $eq 'lua/ai_editor/python' 'C:\Program Files\Python\python.exe' | Out-Null
    Assert-Equal 'equals in other values stay' (Get-ConfValue $eq 'keep') 'a=b'

    $utf16 = Join-Path $tmp 'utf16'
    [System.IO.File]::WriteAllText($utf16, "require `"other`"`n", [System.Text.Encoding]::Unicode)
    Assert-Equal 'utf16 luarc converted' (Update-Luarc $utf16) 'added'
    $bytes = [System.IO.File]::ReadAllBytes($utf16)
    Assert-True 'no utf16 bom left' (-not ($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE))

    Assert-True 'default python is managed' (Test-ManagedPythonCommand 'python3' 'C:\work\.venv\Scripts\python.exe')
    Assert-True 'venv path is managed' (Test-ManagedPythonCommand 'C:\work\.venv\Scripts\python.exe' 'C:\work\.venv\Scripts\python.exe')
    Assert-True 'custom path is not managed' (-not (Test-ManagedPythonCommand 'D:\tools\python.exe' 'C:\work\.venv\Scripts\python.exe'))

$printed = @"
configdir   C:\Users\A B\AppData\Local\darktable
cachedir    C:\Users\A B\AppData\Local\Microsoft\Windows\INetCache\darktable
"@
Assert-Equal 'print-paths configdir' (Get-PrintedConfigDir $printed) 'C:\Users\A B\AppData\Local\darktable'
Assert-True 'print-paths ignores other lines' ($null -eq (Get-PrintedConfigDir "cachedir   C:\nope"))
}
finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Host "FAIL $_" }
    exit 1
}
Write-Host "OK $($PSCommandPath)"
exit 0
