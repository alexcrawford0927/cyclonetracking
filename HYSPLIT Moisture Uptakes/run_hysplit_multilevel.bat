@echo off
setlocal enabledelayedexpansion

REM ==== Base directory ====
set BASEDIR=C:\Users\kripa\Documents\project\storm13_200302

REM ==== Subfolders to loop through ====
for %%F in (height1500 height3000 height5500) do (
    set "WORKDIR=%BASEDIR%\%%F"
    echo Working in: !WORKDIR!
    pushd "!WORKDIR!"

    REM ==== Loop through the list of CONTROL file paths in the current folder ====
    FOR /F "usebackq tokens=* delims=" %%n in ("control_names_200302.txt") do (

        REM ==== Clean old files ====
        IF EXIST "CONTROL" DEL "CONTROL"
        IF EXIST "tdump" DEL "tdump"
        IF EXIST "SETUP.CFG" DEL "SETUP.CFG"

        echo Running trajectory for: %%n

        REM ==== Copy current CONTROL file into place ====
        copy /Y "%%n" "CONTROL" >nul

        REM ==== Run HYSPLIT (hyts_std.exe must read from this folder) ====
        C:\hysplit\exec\hyts_std.exe
    )

    popd
)

endlocal
pause
