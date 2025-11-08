#!/usr/bin/env bash
#
# Morgan v2-0.0.1 - CI/CD Setup Verification Script
#
# This script verifies that all CI/CD components are properly installed.
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Morgan v2-0.0.1 - CI/CD Setup Verification${NC}"
echo ""

# Track results
PASS=0
FAIL=0

check_file() {
    local file=$1
    local desc=$2
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✓${NC} $desc"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} $desc (missing: $file)"
        ((FAIL++))
        return 1
    fi
}

check_executable() {
    local file=$1
    local desc=$2
    if [[ -x "$file" ]]; then
        echo -e "${GREEN}✓${NC} $desc"
        ((PASS++))
        return 0
    else
        echo -e "${RED}✗${NC} $desc (not executable: $file)"
        ((FAIL++))
        return 1
    fi
}

echo "GitHub Actions Workflows:"
check_file ".github/workflows/test.yml" "Test workflow"
check_file ".github/workflows/lint.yml" "Lint workflow"
check_file ".github/workflows/security.yml" "Security workflow"
check_file ".github/workflows/deploy.yml" "Deploy workflow"
echo ""

echo "Pre-commit Configuration:"
check_file ".pre-commit-config.yaml" "Pre-commit config"
check_file ".yamllint.yml" "YAML lint config"
check_file ".secrets.baseline" "Secrets baseline"
echo ""

echo "Automation Scripts:"
check_executable "scripts/setup.sh" "Setup script"
check_executable "scripts/test.sh" "Test script"
check_executable "scripts/lint.sh" "Lint script"
check_executable "scripts/deploy.sh" "Deploy script"
echo ""

echo "Configuration Files:"
check_file "Makefile" "Makefile"
check_file "pyproject.toml" "Python project config"
echo ""

echo "Documentation:"
check_file "docs/CI_CD.md" "CI/CD documentation"
check_file "CI_CD_SETUP_SUMMARY.md" "Setup summary"
echo ""

# Summary
echo "========================================="
echo -e "${GREEN}Passed: $PASS${NC}"
if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}Failed: $FAIL${NC}"
    echo ""
    echo "Some files are missing. Please review the setup."
    exit 1
else
    echo -e "${GREEN}All CI/CD components verified successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. make install        # Install dependencies"
    echo "  2. make install-hooks  # Install pre-commit hooks"
    echo "  3. make test           # Run tests"
    echo "  4. make help           # See all commands"
    exit 0
fi
