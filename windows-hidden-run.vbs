' Run a .cmd file with no console window. Darktable's Lua io.popen
' starts cmd.exe, and that window stays up for the whole Python run.
' Arguments: command file, log file, exit-code file.
Option Explicit
If WScript.Arguments.Count < 3 Then
  WScript.Quit 2
End If
Dim shell, files, code, stream
Set shell = CreateObject("Wscript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
' cmd.exe /c strips quotes in a way that drops the redirect, so the log
' stays from the previous run. /s keeps one outer pair and leaves the rest.
code = shell.Run("cmd.exe /s /c """ & Chr(34) & WScript.Arguments(0) & Chr(34) & " > " & Chr(34) & WScript.Arguments(1) & Chr(34) & " 2>&1""", 0, True)
Set stream = files.CreateTextFile(WScript.Arguments(2), True)
stream.WriteLine CStr(code)
stream.Close
