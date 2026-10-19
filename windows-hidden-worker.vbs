' One hidden process for the whole Darktable session.
' It runs queued .cmd files and creates folders. Nothing here shows a console.
' The queue directory is the only argument.
Option Explicit
If WScript.Arguments.Count < 1 Then
  WScript.Quit 2
End If

Dim shell, files, folder, token
Set shell = CreateObject("Wscript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
folder = WScript.Arguments(0)
Randomize
token = Replace(CStr(Timer), ",", ".") & "-" & CStr(Int(Rnd * 1000000))

Sub MakeDir(path)
  If path = "" Or files.FolderExists(path) Then
    Exit Sub
  End If
  Dim parent
  parent = files.GetParentFolderName(path)
  If parent <> "" And parent <> path Then
    MakeDir parent
  End If
  If Not files.FolderExists(path) Then
    files.CreateFolder path
  End If
End Sub

Sub WriteText(path, text)
  Dim stream
  Set stream = files.CreateTextFile(path, True)
  stream.WriteLine text
  stream.Close
End Sub

Function ReadText(path)
  If Not files.FileExists(path) Then
    ReadText = ""
    Exit Function
  End If
  Dim stream
  Set stream = files.OpenTextFile(path, 1, False)
  If stream.AtEndOfStream Then
    ReadText = ""
  Else
    ReadText = stream.ReadAll
  End If
  stream.Close
End Function

Function Field(text, index)
  Dim parts, item
  parts = Split(Replace(text, vbCrLf, vbLf), vbLf)
  Field = ""
  If index >= 0 And index <= UBound(parts) Then
    item = Replace(parts(index), vbCr, "")
    Field = Trim(item)
  End If
End Function

MakeDir folder
WriteText folder & "\owner.txt", token
WriteText folder & "\ready.txt", "ready"

Do
  If Trim(Replace(ReadText(folder & "\owner.txt"), vbCrLf, "")) <> token Then
    WScript.Quit 0
  End If
  WriteText folder & "\heartbeat.txt", "1"
  If files.FileExists(folder & "\request.txt") Then
    Dim request, kind, code
    request = ReadText(folder & "\request.txt")
    files.DeleteFile folder & "\request.txt", True
    kind = Field(request, 0)
    If kind = "run" Then
      code = shell.Run("cmd.exe /s /c """ & Chr(34) & Field(request, 1) & Chr(34) & " > " & Chr(34) & Field(request, 2) & Chr(34) & " 2>&1""", 0, True)
      WriteText Field(request, 3), CStr(code)
    ElseIf kind = "mkdir" Then
      MakeDir Field(request, 1)
      WriteText Field(request, 2), "0"
    End If
  Else
    WScript.Sleep 40
  End If
Loop
