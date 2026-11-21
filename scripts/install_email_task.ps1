<#
.SYNOPSIS
Registers the 16:00 email task: every weekday at 16:00 New York time it asks the recommendations API to fetch the
day's data and email its lists to RECOMMENDER_EMAIL_TO (scripts/email_recommendations.py).

.DESCRIPTION
The API must be running: scripts/install_task.ps1 registers the task that keeps it up. If the computer is asleep or
off at 16:00, the email goes out as soon as it is available again. The task logs to logs/email.log. Run this again to
update the task, e.g. after moving the repository.

    powershell -ExecutionPolicy Bypass -File scripts/install_email_task.ps1 [-Python <path to pythonw.exe>]
    powershell -ExecutionPolicy Bypass -File scripts/install_email_task.ps1 -Uninstall
#>
param(
    [string]$Python = "$env:USERPROFILE\miniconda3\pythonw.exe",  # Must have the project's packages.
    [string]$TaskName = "ML510 recommendations email",
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed '$TaskName'."
    return
}
if (-not (Test-Path $Python)) {
    throw "No Python at $Python; pass -Python with a pythonw.exe that has the project's packages."
}
$root = Split-Path $PSScriptRoot -Parent
$user = "$env:USERDOMAIN\$env:USERNAME"
# 16:00 New York in local time, as of today.
$newYork = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$today = [datetime]::SpecifyKind([datetime]::Today, "Unspecified")
$at = [TimeZoneInfo]::ConvertTime($today.AddHours(16), $newYork, [TimeZoneInfo]::Local)

$task = @{
    TaskName  = $TaskName
    Action    = New-ScheduledTaskAction -Execute $Python -Argument "`"$root\scripts\email_recommendations.py`"" -WorkingDirectory $root
    Trigger   = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $at
    # Run on battery too, as soon as possible after a missed 16:00, for at most 2 hours, and once at a time.
    Settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2) -MultipleInstances IgnoreNew
    Principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
}
Register-ScheduledTask @task -Force | Out-Null
Write-Host "Registered '$TaskName': weekdays at $($at.ToString('HH:mm')) local time; it logs to $root\logs\email.log."
