#!/bin/bash

echo "=== COMPREHENSIVE PROJECT REORGANIZATION ==="
echo "This script will reorganize your project for better structure and clarity"
echo ""

# --- Step 1: Create New Directory Structure ---
echo "1. Creating new directory structure..."

# Main organization directories
mkdir -p scripts/main
mkdir -p scripts/testing
mkdir -p scripts/utilities
mkdir -p docs/academic
mkdir -p docs/technical
mkdir -p docs/reports
mkdir -p temp/backup

# Consolidate results
mkdir -p results/consolidated/clinical
mkdir -p results/consolidated/mri
mkdir -p results/consolidated/visualizations
mkdir -p results/consolidated/predictions

echo "✅ Created new directory structure"

# --- Step 2: Move and Consolidate Documentation ---
echo ""
echo "2. Consolidating documentation..."

# Move academic content
if [ -f "essay.md" ]; then
    mv essay.md docs/academic/research_paper.md
    echo "  ✅ Moved essay.md → docs/academic/research_paper.md"
fi

# Move technical documentation
if [ -f "PROJECT_COMPLETENESS_REPORT.md" ]; then
    mv PROJECT_COMPLETENESS_REPORT.md docs/technical/project_completeness_report.md
    echo "  ✅ Moved PROJECT_COMPLETENESS_REPORT.md → docs/technical/project_completeness_report.md"
fi

# Move analysis report
if [ -f "results_analysis_report.md" ]; then
    mv results_analysis_report.md docs/reports/results_analysis_report.md
    echo "  ✅ Moved results_analysis_report.md → docs/reports/results_analysis_report.md"
fi

# --- Step 3: Organize Scripts ---
echo ""
echo "3. Organizing scripts..."

# Move main pipeline scripts
if [ -f "run_pipeline.py" ]; then
    mv run_pipeline.py scripts/main/run_pipeline.py
    echo "  ✅ Moved run_pipeline.py → scripts/main/run_pipeline.py"
fi

if [ -f "run_improvements.py" ]; then
    mv run_improvements.py scripts/main/run_improvements.py
    echo "  ✅ Moved run_improvements.py → scripts/main/run_improvements.py"
fi

# Move testing scripts
if [ -f "test_system.py" ]; then
    mv test_system.py scripts/testing/test_system.py
    echo "  ✅ Moved test_system.py → scripts/testing/test_system.py"
fi

if [ -f "test_bias_correction_simple.py" ]; then
    mv test_bias_correction_simple.py scripts/testing/test_bias_correction.py
    echo "  ✅ Moved test_bias_correction_simple.py → scripts/testing/test_bias_correction.py"
fi

# Move utility scripts
if [ -f "combined.py" ]; then
    mv combined.py scripts/utilities/merge_code_files.py
    echo "  ✅ Moved combined.py → scripts/utilities/merge_code_files.py"
fi

# Move reorganization script to utilities
if [ -f "reorganize_project.sh" ]; then
    mv reorganize_project.sh scripts/utilities/reorganize_project_old.sh
    echo "  ✅ Moved reorganize_project.sh → scripts/utilities/reorganize_project_old.sh"
fi

# --- Step 4: Handle Large Files ---
echo ""
echo "4. Handling large files..."

# Backup and remove large merged file
if [ -f "merged.txt" ]; then
    echo "  ⚠️  Large file detected: merged.txt (292KB)"
    echo "  📁 Moving to temp/backup/ for potential future reference"
    mv merged.txt temp/backup/merged_code_backup.txt
    echo "  ✅ Moved merged.txt → temp/backup/merged_code_backup.txt"
fi

# --- Step 5: Consolidate Results Directories ---
echo ""
echo "5. Consolidating results directories..."

# Function to safely move directory contents
move_results_content() {
    local source_dir=$1
    local target_dir=$2
    local description=$3
    
    if [ -d "$source_dir" ]; then
        echo "  📁 Moving $description..."
        mkdir -p "$target_dir"
        mv "$source_dir"/* "$target_dir"/ 2>/dev/null || true
        rmdir "$source_dir" 2>/dev/null || true
        echo "  ✅ Consolidated $source_dir → $target_dir"
    fi
}

# Consolidate clinical results
move_results_content "results/clinical" "results/consolidated/clinical" "clinical results"
move_results_content "results/clinical_model_plots" "results/consolidated/clinical" "clinical model plots"

# Consolidate MRI results
move_results_content "results/mri_analysis" "results/consolidated/mri" "MRI analysis"
move_results_content "results/mri_visualization" "results/consolidated/visualizations" "MRI visualization"
move_results_content "results/mri_visualization_full" "results/consolidated/visualizations" "MRI visualization full"

# Consolidate predictions
move_results_content "results/predictions" "results/consolidated/predictions" "predictions"

# Consolidate PCA/KPCA results
move_results_content "results/pca_kpca" "results/consolidated/visualizations" "PCA/KPCA results"

# --- Step 6: Clean Up System Files ---
echo ""
echo "6. Cleaning up system files..."

# Remove .DS_Store files (macOS system files)
find . -name ".DS_Store" -type f -delete
echo "  ✅ Removed .DS_Store files"

# Remove __pycache__ directories
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✅ Removed __pycache__ directories"

# --- Step 7: Create New README Structure ---
echo ""
echo "7. Creating new project structure documentation..."

cat > PROJECT_STRUCTURE.md << 'EOF'
# Project Structure

## 📁 Directory Organization

### `/docs/` - Documentation
- `/academic/` - Research papers and academic content
- `/technical/` - Technical documentation and reports
- `/reports/` - Analysis reports and results summaries

### `/scripts/` - Executable Scripts
- `/main/` - Main pipeline and execution scripts
- `/testing/` - Test scripts and validation tools
- `/utilities/` - Utility and helper scripts

### `/src/` - Source Code
- `/clinical_data_src/` - Clinical data processing modules
- `/mri_src/` - MRI analysis modules
- `/api/` - API interfaces
- `/utils/` - Utility functions
- `/visualization/` - Visualization modules

### `/data/` - Data Files
- `/raw_lab_data/` - Raw laboratory data
- `/processed_clinical/` - Processed clinical data
- `/288_dicom/` - DICOM files
- `/mri_health/` - Healthy control MRI data
- `/mri_AS/` - AS patient MRI data

### `/results/` - Output Results
- `/consolidated/` - Consolidated results
  - `/clinical/` - Clinical model results
  - `/mri/` - MRI analysis results
  - `/visualizations/` - All visualization outputs
  - `/predictions/` - Model predictions

### `/temp/` - Temporary Files
- `/backup/` - Backup files and archives

## 🚀 Quick Start

1. **Run Main Pipeline**: `python scripts/main/run_pipeline.py`
2. **Run Improvements**: `python scripts/main/run_improvements.py`
3. **Run Tests**: `python scripts/testing/test_system.py`

## 📚 Documentation

- **Research Paper**: `docs/academic/research_paper.md`
- **Technical Report**: `docs/technical/project_completeness_report.md`
- **Results Analysis**: `docs/reports/results_analysis_report.md`
EOF

echo "  ✅ Created PROJECT_STRUCTURE.md"

# --- Step 8: Update Main README ---
echo ""
echo "8. Updating main README..."

# Create a backup of the original README
if [ -f "README.md" ]; then
    cp README.md temp/backup/README_original.md
    echo "  ✅ Backed up original README.md"
fi

# --- Step 9: Final Cleanup ---
echo ""
echo "9. Final cleanup..."

# Remove empty directories
find . -type d -empty -delete 2>/dev/null || true
echo "  ✅ Removed empty directories"

# --- Step 10: Summary ---
echo ""
echo "=== REORGANIZATION COMPLETE ==="
echo ""
echo "📊 Summary of changes:"
echo "  ✅ Consolidated documentation into /docs/"
echo "  ✅ Organized scripts into /scripts/"
echo "  ✅ Consolidated results into /results/consolidated/"
echo "  ✅ Moved large files to /temp/backup/"
echo "  ✅ Cleaned up system files"
echo "  ✅ Created new project structure documentation"
echo ""
echo "🎯 Next steps:"
echo "  1. Review the new structure in PROJECT_STRUCTURE.md"
echo "  2. Update any hardcoded paths in your scripts"
echo "  3. Test the main pipeline: python scripts/main/run_pipeline.py"
echo "  4. Remove /temp/backup/ if everything works correctly"
echo ""
echo "📁 Key new locations:"
echo "  - Main scripts: scripts/main/"
echo "  - Documentation: docs/"
echo "  - Consolidated results: results/consolidated/"
echo "  - Project structure: PROJECT_STRUCTURE.md" 