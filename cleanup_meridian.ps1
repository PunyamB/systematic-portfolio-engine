# cleanup_meridian.ps1
# Meridian + StrategyResearchLab cleanup script (CONSERVATIVE version)
# May 2026
#
# DESIGN PRINCIPLE: Anything that could be useful as an analytical tool,
# data validator, or template for future experiments is KEPT, not deleted.
# Only true artifacts (backups, build outputs, applied one-off fixes) are removed.
#
# USAGE:
#   .\cleanup_meridian.ps1               # DRY-RUN: shows everything that would be moved
#   .\cleanup_meridian.ps1 -Live         # LIVE: moves files to _cleanup_archive/{timestamp}/
#   .\cleanup_meridian.ps1 -Live -Force  # LIVE: skips per-section confirmation prompts
#
# SAFETY:
# - DRY-RUN by default. You see what would happen with NO changes made.
# - LIVE mode MOVES files to _cleanup_archive/ — does NOT delete.
# - You can restore individual files from the archive at any time.
# - After 7+ days, manually delete the archive folder.

param(
    [switch]$Live = $false,
    [switch]$Force = $false
)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
$MeridianRoot = "D:\Projects\SystematicPortfolioEngine"
$ResearchRoot = "D:\Projects\StrategyResearchLab"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ArchiveRoot = Join-Path $MeridianRoot "_cleanup_archive\$Timestamp"

if (-not (Test-Path $MeridianRoot)) {
    Write-Host "ERROR: Meridian root not found: $MeridianRoot" -ForegroundColor Red
    exit 1
}

# ─────────────────────────────────────────────────────────────
# CATEGORIES (paths are relative to MeridianRoot or ResearchRoot)
# ─────────────────────────────────────────────────────────────
#
# REMOVED FROM ORIGINAL CLEANUP (kept for potential future analysis):
#   - compare_portfolio.py  (portfolio comparison tool)
#   - inspect_schemas.py    (parquet schema inspector)
#   - check_data.py, check_stats.py, _quick_test.py (might be analysis helpers)
#   - fetch_exam_data.py    (data-fetching template)
#   - greeks_calc.py        (options Greeks — useful if ever needed)
#   - project_inventory.py  (project scanner)
#   - seed_ic_fast.py, seed_ic_history.py (re-seeding tools)
#   - All sandbox\check_*.py (diagnostic tools for backtest analysis)
#   - All research\check_*, get_*, compare_*, fetch_* scripts (analysis helpers)
#   - run_experiment.py     (experiment template)

# === MERIDIAN — Section A: Patch backups (May 2026 A1-A5c series) ===
$M_A_PatchBackups = @(
    "config\settings.yaml.bak.a1",
    "config\settings.yaml.bak.a2",
    "config\settings.yaml.bak.a3",
    "config\settings.yaml.bak.a5",
    "dashboard\app.py.bak.a1",
    "dashboard\app.py.bak.a2",
    "dashboard\app.py.bak.a5b",
    "execute.py.bak.a1",
    "execute.py.bak.a2",
    "notebooks\backtest_layer2_simulate.py.bak.a5c",
    "notebooks\wf_layer0_calibrate.py.bak.a5c",
    "notebooks\wf_layer2_simulate.py.bak.a5c",
    "optimizer\portfolio_optimizer.py.bak.a3",
    "optimizer\portfolio_optimizer.py.bak.a4",
    "optimizer\portfolio_optimizer.py.bak.a5",
    "pipeline\runner.py.bak.a1",
    "pipeline\runner.py.bak.a2",
    "pipeline\runner.py.bak.a5b",
    "risk\monitor.py.bak.a5b",
    "sandbox\regime_param_search.py.bak.a5c",
    "sandbox\run_backtest.py.bak.a5c",
    "utils\rebalance_calendar.py.bak.a1",
    "utils\rebalance_calendar.py.bak.a2"
)

# === MERIDIAN — Section B: Pure artifacts (definitely no future value) ===
$M_B_PureArtifacts = @(
    "add_exp007.py",            # EXP007 already integrated
    "data_export.csv",          # old data export
    "fix_meridian.py",          # one-off fix, you said fuck it
    "fix_spy.py",               # one-off fix, you said fuck it
    "te_audit.txt",             # TE removal artifact
    "tree.txt",                 # one-off tree dump
    "meridian_tree.txt",        # this session's tree dump
    "research_tree.txt",        # this session's tree dump
    "walk_forward_log.json"     # duplicate of file in research lab
)

# === MERIDIAN — Section C: Build artifacts ===
$M_C_BuildArtifacts = @(
    "meridian_code.zip"
)

# === RESEARCH — Section D: Pure artifacts in research lab ===
$R_D_PureArtifacts = @(
    "fix_spy.py",               # one-off fix
    "check_spy_old.py",         # one-off check on old SPY data
    "debug_files.zip"           # debug dump
)

# === RESEARCH — Section E: EXP008 fix-script artifacts (the indent/fix008 scripts) ===
# Pure one-time syntax fixes. The actual EXP008 run scripts (run_exp008.py and
# run_exp008_all.py) are kept — these are just the broken-and-fixed versions.
$R_E_Exp008FixScripts = @(
    "experiments\exp008_sweep\fix_indent.py",
    "experiments\exp008_sweep\fix008.py",
    "experiments\exp008_sweep\fix008b.py",
    "experiments\exp008_sweep\_run_combo_1.py",
    "experiments\exp008_sweep\_run_combo_2.py",
    "experiments\exp008_sweep\_run_combo_3.py",
    "experiments\exp008_sweep\_run_combo_4.py"
)

# === RESEARCH — Section F: Old backup parquet files ===
# These literally have "backup_2004" in the name, superseded by current versions
$R_F_DataBackups = @(
    "data\raw\breadth_backup_2004.parquet",
    "data\raw\macro_features_backup_2004.parquet"
)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

function Show-Header {
    param([string]$Title, [string]$Color = "Cyan")
    Write-Host ""
    Write-Host ("=" * 75) -ForegroundColor $Color
    Write-Host "  $Title" -ForegroundColor $Color
    Write-Host ("=" * 75) -ForegroundColor $Color
}

function Get-FileSizeKB {
    param([string]$Path)
    if (Test-Path $Path) {
        $size = (Get-Item $Path).Length / 1KB
        return "{0,9:N1} KB" -f $size
    }
    return "   MISSING"
}

function Process-FileList {
    param(
        [string]$SectionName,
        [string]$ProjectRoot,
        [string[]]$Files,
        [bool]$IsLive,
        [bool]$SkipConfirm
    )

    Show-Header $SectionName

    Push-Location $ProjectRoot

    try {
        $existingFiles = $Files | Where-Object { Test-Path $_ }
        $missingFiles = $Files | Where-Object { -not (Test-Path $_) }

        if ($missingFiles.Count -gt 0) {
            Write-Host "[$($missingFiles.Count) file(s) already gone - skipping]" -ForegroundColor DarkGray
        }

        if ($existingFiles.Count -eq 0) {
            Write-Host "Nothing to process in this section." -ForegroundColor Yellow
            return
        }

        Write-Host "Files to process ($($existingFiles.Count)):"
        Write-Host ""
        $totalKB = 0
        foreach ($file in $existingFiles) {
            $sizeStr = Get-FileSizeKB $file
            $sizeNum = (Get-Item $file).Length / 1KB
            $totalKB += $sizeNum
            Write-Host "  $sizeStr  $file"
        }
        Write-Host ""
        Write-Host ("  TOTAL: {0,9:N1} KB ({1} files)" -f $totalKB, $existingFiles.Count) -ForegroundColor Yellow
        Write-Host ""

        if (-not $IsLive) {
            Write-Host "[DRY-RUN] No action taken." -ForegroundColor Yellow
            return
        }

        if (-not $SkipConfirm) {
            $response = Read-Host "Move these files to archive? (y/N)"
            if ($response -ne "y" -and $response -ne "Y") {
                Write-Host "Section skipped." -ForegroundColor Yellow
                return
            }
        }

        $projectName = Split-Path $ProjectRoot -Leaf
        $sectionArchive = Join-Path $ArchiveRoot ($projectName + "\" + ($SectionName -replace "[^A-Za-z0-9]", "_"))
        if (-not (Test-Path $sectionArchive)) {
            New-Item -Path $sectionArchive -ItemType Directory -Force | Out-Null
        }

        foreach ($file in $existingFiles) {
            try {
                $destDir = Join-Path $sectionArchive (Split-Path $file -Parent)
                if ($destDir -and -not (Test-Path $destDir)) {
                    New-Item -Path $destDir -ItemType Directory -Force | Out-Null
                }
                $destPath = Join-Path $sectionArchive $file
                Move-Item -Path $file -Destination $destPath -Force
                Write-Host "  MOVED: $file" -ForegroundColor Green
            } catch {
                Write-Host "  FAILED: $file - $_" -ForegroundColor Red
            }
        }
    } finally {
        Pop-Location
    }
}

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

Show-Header "MERIDIAN + STRATEGYRESEARCHLAB CLEANUP (CONSERVATIVE)" "Magenta"
Write-Host "Meridian root : $MeridianRoot"
Write-Host "Research root : $ResearchRoot"
Write-Host "Mode          : $(if ($Live) { 'LIVE (files moved to archive)' } else { 'DRY-RUN (no changes)' })"
if ($Live) {
    Write-Host "Archive at    : $ArchiveRoot"
}
Write-Host ""
Write-Host "Design principle: anything potentially useful for future analysis is KEPT."
Write-Host "Only true artifacts (backups, build outputs, applied fixes) are removed."
Write-Host ""

if (-not $Live) {
    Write-Host "Re-run with: .\cleanup_meridian.ps1 -Live" -ForegroundColor Yellow
    Write-Host "Add -Force to skip per-section prompts" -ForegroundColor Yellow
}

Show-Header "GIT STATUS - MERIDIAN" "Yellow"
Push-Location $MeridianRoot
$gitStatus = git status --porcelain 2>$null
if ($LASTEXITCODE -eq 0) {
    if ($gitStatus) {
        Write-Host "WARNING: Uncommitted changes in Meridian." -ForegroundColor Yellow
        $gitStatus | Select-Object -First 10 | ForEach-Object { Write-Host "  $_" }
        if ($gitStatus.Count -gt 10) {
            Write-Host "  ... and $($gitStatus.Count - 10) more" -ForegroundColor DarkGray
        }
        if ($Live -and -not $Force) {
            Write-Host ""
            $proceed = Read-Host "Proceed anyway? (y/N)"
            if ($proceed -ne "y" -and $proceed -ne "Y") {
                Write-Host "Aborted." -ForegroundColor Red
                Pop-Location
                exit 0
            }
        }
    } else {
        Write-Host "Git working tree clean." -ForegroundColor Green
    }
} else {
    Write-Host "Not a git repository." -ForegroundColor DarkGray
}
Pop-Location

# Meridian sections
Process-FileList -SectionName "[M-A] Patch Backups (May 2026 A1-A5c series)" `
                 -ProjectRoot $MeridianRoot -Files $M_A_PatchBackups `
                 -IsLive $Live -SkipConfirm $Force

Process-FileList -SectionName "[M-B] Pure Artifacts (definitely no future value)" `
                 -ProjectRoot $MeridianRoot -Files $M_B_PureArtifacts `
                 -IsLive $Live -SkipConfirm $Force

Process-FileList -SectionName "[M-C] Build Artifacts (zipped backups)" `
                 -ProjectRoot $MeridianRoot -Files $M_C_BuildArtifacts `
                 -IsLive $Live -SkipConfirm $Force

# Research sections
if (-not (Test-Path $ResearchRoot)) {
    Write-Host ""
    Write-Host "Research root not found - skipping research lab cleanup." -ForegroundColor DarkGray
} else {
    Process-FileList -SectionName "[R-D] Research Pure Artifacts" `
                     -ProjectRoot $ResearchRoot -Files $R_D_PureArtifacts `
                     -IsLive $Live -SkipConfirm $Force

    Process-FileList -SectionName "[R-E] EXP008 Fix-Script + Combo Artifacts" `
                     -ProjectRoot $ResearchRoot -Files $R_E_Exp008FixScripts `
                     -IsLive $Live -SkipConfirm $Force

    Process-FileList -SectionName "[R-F] Old Backup Parquet Files (literal *_backup_2004*)" `
                     -ProjectRoot $ResearchRoot -Files $R_F_DataBackups `
                     -IsLive $Live -SkipConfirm $Force
}

Show-Header "DONE" "Green"

if ($Live) {
    if (Test-Path $ArchiveRoot) {
        $archivedFiles = Get-ChildItem -Recurse -File $ArchiveRoot
        $archivedCount = $archivedFiles.Count
        $archivedSize = ($archivedFiles | Measure-Object -Property Length -Sum).Sum / 1KB
        Write-Host ("Archived {0} files ({1:N1} KB) to:" -f $archivedCount, $archivedSize)
        Write-Host "  $ArchiveRoot"
        Write-Host ""
        Write-Host "If everything looks good after a few days, delete the archive:"
        Write-Host "  Remove-Item -Recurse -Force '$ArchiveRoot'" -ForegroundColor DarkGray
        Write-Host ""
        Write-Host "If you need to restore, files are organized by section under that path."
    } else {
        Write-Host "No files were moved (all sections skipped or empty)."
    }
} else {
    Write-Host "DRY-RUN complete. No changes made." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Next step: review the lists above carefully, then run:"
    Write-Host "  .\cleanup_meridian.ps1 -Live" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "You will be prompted before each section so you can skip individual sections."
}
Write-Host ""
