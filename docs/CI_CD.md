# Morgan v2-0.0.1 - CI/CD Documentation

This document describes the complete CI/CD pipeline infrastructure for Morgan v2-0.0.1.

## Table of Contents

1. [Overview](#overview)
2. [GitHub Actions Workflows](#github-actions-workflows)
3. [Pre-commit Hooks](#pre-commit-hooks)
4. [Automation Scripts](#automation-scripts)
5. [Makefile Commands](#makefile-commands)
6. [Setup and Configuration](#setup-and-configuration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Morgan's CI/CD infrastructure provides:

- **Automated Testing** - Unit, integration, and service tests
- **Code Quality** - Linting, formatting, and type checking
- **Security Scanning** - Dependency, secret, and vulnerability scanning
- **Automated Deployment** - Staging and production deployments
- **Pre-commit Hooks** - Local quality checks before commits
- **Automation Scripts** - One-command setup, testing, and deployment
- **Makefile** - Simple commands for common operations
- **NetBird VPN Integration** - Automatic VPN setup for Nexus registry access in CI/CD

### Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         CI/CD Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Local Development                                              │
│  ├── Pre-commit Hooks (format, lint, type-check)               │
│  ├── Makefile Commands (make test, make lint, make deploy)     │
│  └── Scripts (setup.sh, test.sh, lint.sh, deploy.sh)           │
│                                                                 │
│  GitHub Actions (Triggered on PR/Push)                          │
│  ├── Test Workflow (unit, integration, docker)                 │
│  ├── Lint Workflow (black, ruff, mypy, yamllint)               │
│  ├── Security Workflow (bandit, safety, trivy, codeql)         │
│  └── Deploy Workflow (build, push, deploy, health-check)       │
│                                                                 │
│  Deployment Environments                                        │
│  ├── Local (Docker Compose)                                    │
│  ├── Staging (staging.lazarev.cloud)                           │
│  └── Production (morgan.lazarev.cloud)                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## GitHub Actions Workflows

### 1. Test Workflow (`.github/workflows/test.yml`)

**Triggers:**
- Pull requests to `main`, `develop`, `claude/**`
- Pushes to `main`, `develop`
- Manual dispatch

**Jobs:**
- **test** - Run unit tests on Python 3.11 and 3.12
  - Automatically setup NetBird VPN for Nexus access
  - Install dependencies with uv
  - Run pytest with coverage
  - Upload coverage to Codecov
  - Archive test results

- **integration-test** - Run integration tests
  - Start PostgreSQL, Redis, Qdrant services
  - Automatically setup NetBird VPN for Nexus access
  - Run integration tests
  - Only runs on non-draft PRs

- **docker-build-test** - Test Docker builds
  - Automatically setup NetBird VPN for Nexus access
  - Build all service images
  - Validate docker-compose.yml
  - Cache builds for faster runs

- **summary** - Aggregate results
  - Fail if any job fails
  - Report overall status

**Note**: All workflows automatically configure NetBird VPN before dependency installation using the `NETBIRD_SETUP_KEY` secret. This enables access to the private Nexus repository.

**Usage:**
```bash
# Automatically runs on PR creation
# Or trigger manually:
gh workflow run test.yml
```

---

### 2. Lint Workflow (`.github/workflows/lint.yml`)

**Triggers:**
- Pull requests
- Pushes to main branches
- Manual dispatch

**Jobs:**
- **lint** - Python code quality
  - Black (formatting check)
  - isort (import sorting)
  - Ruff (fast linting)
  - mypy (type checking)

- **format-check** - Formatting validation
  - Check if code needs formatting
  - Comment on PR if formatting needed

- **yaml-lint** - YAML validation
  - Check config files
  - Validate GitHub Actions syntax

- **dockerfile-lint** - Dockerfile quality
  - Run hadolint on all Dockerfiles
  - Check best practices

- **shell-check** - Shell script validation
  - Run shellcheck on all scripts
  - Ensure POSIX compliance

- **complexity-check** - Code complexity
  - Cyclomatic complexity
  - Maintainability index

**Usage:**
```bash
# Runs automatically on PRs
# Or manually:
make lint
./scripts/lint.sh --all
```

---

### 3. Security Workflow (`.github/workflows/security.yml`)

**Triggers:**
- Pull requests
- Pushes to main branches
- Daily at 2 AM UTC (scheduled)
- Manual dispatch

**Jobs:**
- **dependency-check** - Dependency security
  - Safety (Python package vulnerabilities)
  - pip-audit (audit Python dependencies)

- **secret-scanning** - Secret detection
  - TruffleHog (scan for secrets in commits)
  - GitLeaks (detect hardcoded secrets)

- **code-security** - Static analysis
  - Bandit (Python security issues)
  - Generate security reports

- **docker-security** - Container scanning
  - Trivy (vulnerability scanner)
  - Scan all service images
  - Upload results to GitHub Security

- **codeql** - Advanced analysis
  - GitHub CodeQL analysis
  - Security and quality queries

- **dependency-review** - PR dependency review
  - Check new dependencies
  - Fail on moderate+ severity

- **license-check** - License compliance
  - Check dependency licenses
  - Fail on GPL/LGPL licenses

**Usage:**
```bash
# Runs automatically
# Or trigger manually:
gh workflow run security.yml
```

---

### 4. Deploy Workflow (`.github/workflows/deploy.yml`)

**Triggers:**
- Push to `main`, `staging`
- Tag push (v*)
- Manual dispatch with environment selection

**Jobs:**
- **build-and-push** - Build images
  - Build all service images
  - Tag with version, branch, SHA
  - Push to Harbor registry
  - Use build cache for speed

- **deploy-staging** - Deploy to staging
  - SSH to staging server
  - Pull latest code
  - Update containers
  - Health checks
  - Deploy status notification

- **deploy-production** - Deploy to production
  - Require manual approval
  - Create database backup
  - Deploy with zero-downtime
  - Run migrations
  - Health checks with retries
  - Rollback on failure
  - Create GitHub release

- **smoke-test** - Post-deployment tests
  - Test all health endpoints
  - Basic API functionality tests
  - Service version checks

- **notify** - Notifications
  - Slack notifications
  - Deployment status updates

**Usage:**
```bash
# Automatic on main/staging push
# Manual deployment:
gh workflow run deploy.yml -f environment=staging
gh workflow run deploy.yml -f environment=production -f version=v2.0.0

# Or use scripts:
make deploy-staging
make deploy-production
```

---

## Pre-commit Hooks

### Configuration (`.pre-commit-config.yaml`)

Pre-commit hooks run automatically before each commit to ensure code quality.

**Hooks included:**
1. **General file checks** - Trailing whitespace, EOF, merge conflicts
2. **Black** - Python code formatting
3. **isort** - Import sorting
4. **Ruff** - Fast Python linting
5. **mypy** - Type checking
6. **Bandit** - Security checks
7. **yamllint** - YAML validation
8. **hadolint** - Dockerfile linting
9. **shellcheck** - Shell script validation
10. **markdownlint** - Markdown formatting
11. **detect-secrets** - Secret detection
12. **pydocstyle** - Docstring validation

### Installation

```bash
# Install pre-commit hooks
make install-hooks

# Or manually:
pre-commit install
pre-commit install --hook-type commit-msg
```

### Usage

```bash
# Automatically runs on git commit
git commit -m "Your message"

# Run manually on all files
pre-commit run --all-files

# Run specific hook
pre-commit run black --all-files

# Skip hooks (use sparingly)
git commit --no-verify -m "Emergency fix"

# Update hooks
pre-commit autoupdate
```

---

## Automation Scripts

### 1. Setup Script (`scripts/setup.sh`)

Complete one-command setup of development environment.

**Features:**
- Check system requirements
- Install uv package manager
- Create virtual environment
- Install all dependencies
- Setup pre-commit hooks
- Create environment files
- Build Docker images
- Initialize databases
- Run basic tests

**Usage:**
```bash
# Full setup
./scripts/setup.sh

# Skip specific steps
./scripts/setup.sh --skip-docker --skip-hooks

# Production setup (minimal dev tools)
./scripts/setup.sh --production

# Or use Makefile:
make install
make quickstart
```

**Options:**
- `--skip-uv` - Skip uv installation
- `--skip-docker` - Skip Docker setup
- `--skip-hooks` - Skip pre-commit hooks
- `--production` - Production mode
- `--help` - Show help

---

### 2. Test Script (`scripts/test.sh`)

Comprehensive test runner with multiple modes.

**Features:**
- Unit tests
- Integration tests
- Coverage reports
- Watch mode
- Parallel execution
- Docker tests
- Service health checks

**Usage:**
```bash
# Run all tests
./scripts/test.sh --all

# Unit tests only
./scripts/test.sh --unit

# Integration tests
./scripts/test.sh --integration

# With coverage
./scripts/test.sh --all --coverage

# Watch mode
./scripts/test.sh --watch

# Fast mode (skip slow tests)
./scripts/test.sh --fast

# Re-run failed tests
./scripts/test.sh --failed

# Parallel execution
./scripts/test.sh --parallel

# Docker tests
./scripts/test.sh --docker

# Service health checks
./scripts/test.sh --services

# Or use Makefile:
make test
make test-unit
make test-integration
make test-coverage
```

**Options:**
- `--unit` - Unit tests only
- `--integration` - Integration tests only
- `--all` - All tests
- `--coverage` - Generate coverage
- `--watch` - Watch mode
- `--verbose` - Verbose output
- `--fast` - Skip slow tests
- `--parallel` - Parallel execution
- `--failed` - Re-run failed only
- `--docker` - Run in Docker
- `--services` - Test services
- `--help` - Show help

---

### 3. Lint Script (`scripts/lint.sh`)

Comprehensive code quality checks.

**Features:**
- Black formatting
- isort import sorting
- Ruff linting
- mypy type checking
- Bandit security scanning
- Safety vulnerability checks
- Code complexity analysis
- YAML linting
- Dockerfile linting
- Shell script checking

**Usage:**
```bash
# Check all
./scripts/lint.sh --check

# Auto-fix issues
./scripts/lint.sh --fix

# Format code
./scripts/lint.sh --format

# Type checking
./scripts/lint.sh --type

# Security checks
./scripts/lint.sh --security

# All checks
./scripts/lint.sh --all

# Fast mode
./scripts/lint.sh --fast

# CI mode (fail on errors)
./scripts/lint.sh --ci

# Or use Makefile:
make lint
make lint-fix
make format
make type-check
make security
```

**Options:**
- `--fix` - Auto-fix issues
- `--check` - Check only (default)
- `--format` - Format code
- `--type` - Type checking
- `--security` - Security checks
- `--complexity` - Complexity check
- `--all` - All checks
- `--fast` - Skip slow checks
- `--ci` - CI mode
- `--help` - Show help

---

### 4. Deploy Script (`scripts/deploy.sh`)

Deployment automation with rollback support.

**Features:**
- Local deployment
- Staging deployment
- Production deployment
- Automatic backups
- Health checks
- Rollback capability
- Version management

**Usage:**
```bash
# Deploy locally
./scripts/deploy.sh local

# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh production

# Deploy specific version
./scripts/deploy.sh production --version v2.0.0

# With backup
./scripts/deploy.sh production --backup

# Dry run (preview)
./scripts/deploy.sh production --dry-run

# Rollback
./scripts/deploy.sh production --rollback

# Health check only
./scripts/deploy.sh production --health-check

# Force deployment
./scripts/deploy.sh staging --force

# Or use Makefile:
make deploy-local
make deploy-staging
make deploy-production
make rollback-production
make health-production
```

**Options:**
- `--version VERSION` - Deploy specific version
- `--backup` - Create backup
- `--no-backup` - Skip backup
- `--rollback` - Rollback deployment
- `--health-check` - Health check only
- `--dry-run` - Preview only
- `--force` - Skip confirmations
- `--help` - Show help

---

## Makefile Commands

### Quick Reference

```bash
# Setup
make install          # Install dependencies
make install-dev      # Install dev dependencies
make install-hooks    # Install pre-commit hooks

# Testing
make test             # Run all tests
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-coverage    # With coverage report
make test-watch       # Watch mode

# Code Quality
make lint             # Run linters
make lint-fix         # Auto-fix issues
make format           # Format code
make type-check       # Type checking
make security         # Security checks

# Docker
make docker-build     # Build images
make docker-up        # Start services
make docker-down      # Stop services
make docker-restart   # Restart services
make docker-logs      # View logs
make docker-clean     # Clean resources

# Development
make run              # Start services
make stop             # Stop services
make logs             # View logs
make shell            # Open shell in core
make health           # Check service health

# Database
make db-migrate       # Run migrations
make db-backup        # Backup database
make db-restore       # Restore database
make db-reset         # Reset database

# Deployment
make deploy-local     # Deploy locally
make deploy-staging   # Deploy to staging
make deploy-production # Deploy to production

# Cleaning
make clean            # Clean temp files
make clean-all        # Clean everything

# Quick Start
make quickstart       # Complete setup
make help             # Show all commands
```

---

## Setup and Configuration

### Initial Setup

1. **Clone repository:**
   ```bash
   git clone <repository-url>
   cd morgan
   ```

2. **Run setup:**
   ```bash
   make install
   # Or
   ./scripts/setup.sh
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Install pre-commit hooks:**
   ```bash
   make install-hooks
   ```

5. **Start services:**
   ```bash
   make run
   ```

### GitHub Actions Setup

1. **Required secrets:**
   ```bash
   CODECOV_TOKEN              # Codecov upload token
   HARBOR_USERNAME            # Harbor registry username
   HARBOR_PASSWORD            # Harbor registry password
   NETBIRD_SETUP_KEY          # NetBird VPN setup key (for Nexus access)
   STAGING_SSH_KEY            # Staging server SSH key
   STAGING_USER               # Staging server user
   STAGING_HOST               # Staging server hostname
   PRODUCTION_SSH_KEY         # Production server SSH key
   PRODUCTION_USER            # Production server user
   PRODUCTION_HOST            # Production server hostname
   SLACK_WEBHOOK_URL          # Slack notifications
   ```

2. **Add secrets:**
   ```bash
   # Via GitHub web UI:
   Settings → Secrets and variables → Actions → New repository secret

   # Or using gh CLI:
   gh secret set CODECOV_TOKEN < token.txt
   gh secret set HARBOR_USERNAME
   gh secret set HARBOR_PASSWORD
   ```

3. **Enable workflows:**
   - Go to Actions tab
   - Enable workflows if disabled
   - Configure required approvals for production

### Pre-commit Setup

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test installation
pre-commit run --all-files

# Update hooks
pre-commit autoupdate
```

---

## Best Practices

### Development Workflow

1. **Before starting work:**
   ```bash
   git checkout -b feature/your-feature
   make install
   make run
   ```

2. **During development:**
   ```bash
   # Format code regularly
   make format

   # Run tests
   make test-unit

   # Check linting
   make lint
   ```

3. **Before committing:**
   ```bash
   # Pre-commit hooks run automatically
   git add .
   git commit -m "Your message"

   # Or run manually first
   make pre-commit
   ```

4. **Before pushing:**
   ```bash
   # Run full test suite
   make test

   # Run all linters
   make lint-all

   # Run security checks
   make security
   ```

5. **Create PR:**
   - All CI checks should pass
   - Address any linting issues
   - Review security scan results

### CI/CD Best Practices

1. **Test locally first:**
   ```bash
   make ci  # Run all CI checks locally
   ```

2. **Use feature branches:**
   - Branch from `develop`
   - Name: `feature/`, `fix/`, `refactor/`
   - Create PR to `develop`

3. **Keep commits clean:**
   - Use conventional commits
   - One logical change per commit
   - Write clear commit messages

4. **Review before merge:**
   - All CI checks passing
   - Code reviewed
   - Documentation updated

5. **Deployment workflow:**
   ```text
   feature → develop → staging → main → production
   ```

---

## Troubleshooting

### Common Issues

#### Pre-commit hooks fail

**Problem:** Pre-commit hooks fail on commit

**Solution:**
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Clear cache
pre-commit clean

# Reinstall
pre-commit uninstall
pre-commit install

# Run manually to see errors
pre-commit run --all-files
```

#### Tests fail in CI but pass locally

**Problem:** Tests pass locally but fail in GitHub Actions

**Solution:**
```bash
# Run tests in Docker (same as CI)
make test-docker

# Check environment variables
# Ensure CI environment matches local

# Check for flaky tests
make test --failed --failed --failed
```

#### Docker build failures

**Problem:** Docker build fails in CI

**Solution:**
```bash
# Test build locally
make docker-build-no-cache

# Check Dockerfile syntax
hadolint core/Dockerfile

# Validate docker-compose
docker compose config
```

#### Linter conflicts

**Problem:** Different linters give conflicting suggestions

**Solution:**
```bash
# Black and isort are configured to work together
# Run in correct order:
make format  # Black + isort
make lint    # Then other linters
```

#### Security scan false positives

**Problem:** Security scanner reports false positives

**Solution:**
```bash
# Review the report
cat safety-report.json

# Add exceptions if needed (carefully)
# Update bandit config in pyproject.toml
# Update safety ignore list
```

### Getting Help

1. **Check logs:**
   ```bash
   # Local logs
   make logs
   docker compose logs -f service-name

   # CI logs
   # View on GitHub Actions tab
   ```

2. **Run in verbose mode:**
   ```bash
   ./scripts/test.sh --verbose
   ./scripts/lint.sh --verbose
   ```

3. **Clean and retry:**
   ```bash
   make clean
   make docker-clean
   make install
   ```

4. **Check documentation:**
   - README.md - Project overview
   - DEVELOPMENT.md - Development guide
   - TROUBLESHOOTING.md - Common issues

---

## Additional Resources

### Documentation
- [Quick Start Guide](getting-started/QUICK_START.md)
- [Development Guide](getting-started/DEVELOPMENT.md)
- [API Documentation](architecture/API.md)
- [Deployment Guide](deployment/DEPLOYMENT.md)

### External Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pre-commit Documentation](https://pre-commit.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Python Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)

---

## Summary

Morgan v2-0.0.1 includes a comprehensive CI/CD infrastructure that:

✅ **Automates testing** - Unit, integration, and service tests
✅ **Ensures code quality** - Formatting, linting, type checking
✅ **Enforces security** - Scanning for vulnerabilities and secrets
✅ **Streamlines deployment** - Automated staging and production deploys
✅ **Provides local tools** - Scripts and Makefile for development
✅ **Maintains reliability** - Health checks and rollback capabilities

**Quick start:**
```bash
make quickstart  # Complete setup and start
make help        # See all available commands
```

**For daily development:**
```bash
make run         # Start services
make test        # Run tests
make lint        # Check code quality
make format      # Format code
```

**For deployment:**
```bash
make deploy-staging      # Deploy to staging
make deploy-production   # Deploy to production
```

---

**Last Updated:** 2025-11-08
**Version:** v2-0.0.1
