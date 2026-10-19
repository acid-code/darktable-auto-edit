' Darktable's program is a console app, so a normal start keeps a terminal
' open for the whole session. This starts the same program with no console.
Option Explicit

Dim exe, work, locator, service, process, startup, result, pid
exe = "C:\Program Files\darktable\bin\darktable.exe"
work = "C:\Program Files\darktable"

Set locator = CreateObject("WbemScripting.SWbemLocator")
Set service = locator.ConnectServer(".", "root\cimv2")
Set process = service.Get("Win32_Process")
Set startup = service.Get("Win32_ProcessStartup").SpawnInstance_()
startup.ShowWindow = 0
result = process.Create("""" & exe & """", work, startup, pid)
If result <> 0 Then
  WScript.Echo "Darktable did not start (" & CStr(result) & ")."
  WScript.Quit result
End If
