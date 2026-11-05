<#
.SYNOPSIS
Registers a Windows scheduled task that keeps the recommendations API running in the background, so its
weekday 16:00 New York refresh fetches the data and emails the lists without starting the API by hand.

.DESCRIPTION
The task runs scripts/serve.py with pythonw (no window) when you log on, and again at 15:55 New York time on
weekdays in case the API stopped; serve.py exits at once if an API already listens on its port. Run this
again to update the task, e.g. after moving the repository.

    powershell -ExecutionPolicy Bypass -File scripts/install_task.ps1 [-Python <path to pythonw.exe>]
    powershell -ExecutionPolicy Bypass -File scripts/install_task.ps1 -Uninstall
#>
param(
    [string]$Python = "$env:USERPROFILE\miniconda3\pythonw.exe",  # Must have the project's packages.
    [string]$TaskName = "ML510 stock recommender",
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"

if ($Uninstall) {
    Stop-ScheduledTask -TaskName $TaskName  # Also stops the API.
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed '$TaskName'."
    return
}
if (-not (Test-Path $Python)) {
    throw "No Python at $Python; pass -Python with a pythonw.exe that has the project's packages."
}
$root = Split-Path $PSScriptRoot -Parent
$user = "$env:USERDOMAIN\$env:USERNAME"
# 15:55 New York in local time, as of today.
$newYork = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$today = [datetime]::SpecifyKind([datetime]::Today, "Unspecified")
$restartAt = [TimeZoneInfo]::ConvertTime($today.AddHours(15).AddMinutes(55), $newYork, [TimeZoneInfo]::Local)

$task = @{
    TaskName  = $TaskName
    Action    = New-ScheduledTaskAction -Execute $Python -Argument "`"$root\scripts\serve.py`"" -WorkingDirectory $root
    Trigger   = @(
        New-ScheduledTaskTrigger -AtLogOn -User $user
        New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $restartAt
    )
    # Run on battery too, with no time limit, and once: a start while the API runs is ignored.
    Settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    Principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
}
Register-ScheduledTask @task -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Registered and started '$TaskName'; the API logs to $root\logs\api.log."
