Option Explicit
' Starts windows-hidden-run.vbs and returns at once.
' The caller is cmd.exe, and cmd stays on screen until this script exits.
' The real edit keeps running in the hidden script.
If WScript.Arguments.Count < 4 Then
  WScript.Quit 2
End If
Dim shell, command, index
Set shell = CreateObject("Wscript.Shell")
command = "wscript //nologo //b """ & WScript.Arguments(0) & """"
For index = 1 To 3
  command = command & " """ & WScript.Arguments(index) & """"
Next
shell.Run command, 0, False
