# Morgan v2-0.0.1 Makefile
# Common development and deployment commands
#
# Usage:
#   make help          - Show this help message
#   make install       - Install all dependencies
#   make test          - Run tests
#   make lint          - Run linters
#   make format        - Format code
#   make run           - Start all services
#

.PHONY: help install test lint format run clean docker-build docker-up docker-down deploy stop restart logs ci

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

# Project configuration
PYTHON := python3
VENV := .venv
VENV_BIN := $(VENV)/bin
UV := $(shell command -v uv 2> /dev/null)

##@ General

help: ## Display this help message
	@echo "$(BLUE)Morgan v2-0.0.1 - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage: make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Setup & Installation

install: ## Install all dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@./scripts/setup.sh
	@echo "$(GREEN)Installation complete!$(NC)"

install-dev: ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	@if [ -z "$(UV)" ]; then \
		$(VENV_BIN)/pip install pytest pytest-cov pytest-asyncio pytest-mock httpx \
			black isort ruff mypy types-PyYAML types-redis \
			pre-commit bandit safety pip-licenses radon; \
	else \
		uv pip install pytest pytest-cov pytest-asyncio pytest-mock httpx \
			black isort ruff mypy types-PyYAML types-redis \
			pre-commit bandit safety pip-licenses radon; \
	fi
	@echo "$(GREEN)Development dependencies installed!$(NC)"

install-hooks: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(NC)"
	@$(VENV_BIN)/pre-commit install
	@$(VENV_BIN)/pre-commit install --hook-type commit-msg
	@echo "$(GREEN)Pre-commit hooks installed!$(NC)"

update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	@if [ -z "$(UV)" ]; then \
		$(VENV_BIN)/pip install --upgrade pip; \
		$(VENV_BIN)/pip install --upgrade -r requirements-base.txt; \
	else \
		uv pip install --upgrade -r requirements-base.txt; \
	fi
	@echo "$(GREEN)Dependencies updated!$(NC)"

##@ Testing

test: ## Run all tests
	@./scripts/test.sh --all

test-unit: ## Run unit tests only
	@./scripts/test.sh --unit

test-integration: ## Run integration tests only
	@./scripts/test.sh --integration

test-coverage: ## Run tests with coverage report
	@./scripts/test.sh --all --coverage

test-watch: ## Run tests in watch mode
	@./scripts/test.sh --watch

test-fast: ## Run tests (skip slow ones)
	@./scripts/test.sh --fast

test-failed: ## Re-run only failed tests
	@./scripts/test.sh --failed

test-docker: ## Run tests in Docker
	@./scripts/test.sh --docker

test-services: ## Test all microservices health
	@./scripts/test.sh --services

##@ Code Quality

lint: ## Run all linters
	@./scripts/lint.sh --check

lint-fix: ## Run linters and auto-fix issues
	@./scripts/lint.sh --fix

format: ## Format code with Black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@./scripts/lint.sh --fix --format
	@echo "$(GREEN)Code formatted!$(NC)"

type-check: ## Run type checking with mypy
	@./scripts/lint.sh --type

security: ## Run security checks
	@./scripts/lint.sh --security

complexity: ## Check code complexity
	@./scripts/lint.sh --complexity

lint-all: ## Run all linters and checks
	@./scripts/lint.sh --all

pre-commit: ## Run pre-commit on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	@$(VENV_BIN)/pre-commit run --all-files
	@echo "$(GREEN)Pre-commit checks complete!$(NC)"

##@ Docker Operations

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	@docker compose build --pull
	@echo "$(GREEN)Docker images built!$(NC)"

docker-build-no-cache: ## Build Docker images without cache
	@echo "$(BLUE)Building Docker images (no cache)...$(NC)"
	@docker compose build --no-cache --pull
	@echo "$(GREEN)Docker images built!$(NC)"

docker-up: ## Start all Docker services
	@echo "$(BLUE)Starting Docker services...$(NC)"
	@docker compose up -d
	@echo "$(GREEN)Services started!$(NC)"
	@echo ""
	@echo "Services available at:"
	@echo "  - Core:  http://localhost:8000"
	@echo "  - LLM:   http://localhost:8001"
	@echo "  - TTS:   http://localhost:8002"
	@echo "  - STT:   http://localhost:8003"

docker-down: ## Stop all Docker services
	@echo "$(BLUE)Stopping Docker services...$(NC)"
	@docker compose down
	@echo "$(GREEN)Services stopped!$(NC)"

docker-restart: ## Restart all Docker services
	@make docker-down
	@make docker-up

docker-logs: ## Show Docker logs
	@docker compose logs -f

docker-logs-core: ## Show Core service logs
	@docker compose logs -f core

docker-logs-llm: ## Show LLM service logs
	@docker compose logs -f llm-service

docker-logs-tts: ## Show TTS service logs
	@docker compose logs -f tts-service

docker-logs-stt: ## Show STT service logs
	@docker compose logs -f stt-service

docker-ps: ## Show running Docker containers
	@docker compose ps

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	@docker compose down -v
	@docker system prune -af
	@echo "$(GREEN)Docker cleaned!$(NC)"

##@ Development

run: docker-up ## Start all services (alias for docker-up)

stop: docker-down ## Stop all services (alias for docker-down)

restart: docker-restart ## Restart all services (alias for docker-restart)

logs: docker-logs ## Show logs (alias for docker-logs)

shell: ## Open shell in core container
	@docker compose exec core bash

shell-llm: ## Open shell in LLM container
	@docker compose exec llm-service bash

shell-tts: ## Open shell in TTS container
	@docker compose exec tts-service bash

shell-stt: ## Open shell in STT container
	@docker compose exec stt-service bash

db-shell: ## Open PostgreSQL shell
	@docker compose exec postgres psql -U morgan -d morgan

redis-cli: ## Open Redis CLI
	@docker compose exec redis redis-cli

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -f http://localhost:8000/health && echo "$(GREEN)✓ Core$(NC)" || echo "$(YELLOW)✗ Core$(NC)"
	@curl -f http://localhost:8001/health && echo "$(GREEN)✓ LLM$(NC)" || echo "$(YELLOW)✗ LLM$(NC)"
	@curl -f http://localhost:8002/health && echo "$(GREEN)✓ TTS$(NC)" || echo "$(YELLOW)✗ TTS$(NC)"
	@curl -f http://localhost:8003/health && echo "$(GREEN)✓ STT$(NC)" || echo "$(YELLOW)✗ STT$(NC)"

watch: ## Watch for changes and restart services
	@echo "$(BLUE)Watching for changes...$(NC)"
	@docker compose watch

##@ Database

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	@docker compose exec core python -m alembic upgrade head
	@echo "$(GREEN)Migrations complete!$(NC)"

db-rollback: ## Rollback last database migration
	@echo "$(BLUE)Rolling back last migration...$(NC)"
	@docker compose exec core python -m alembic downgrade -1
	@echo "$(GREEN)Rollback complete!$(NC)"

db-backup: ## Backup database
	@echo "$(BLUE)Backing up database...$(NC)"
	@mkdir -p backups
	@docker compose exec -T postgres pg_dump -U morgan morgan > backups/backup_$$(date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)Database backed up!$(NC)"

db-restore: ## Restore database from latest backup (use BACKUP_FILE=path to specify)
	@echo "$(BLUE)Restoring database...$(NC)"
	@if [ -z "$(BACKUP_FILE)" ]; then \
		LATEST=$$(ls -t backups/*.sql 2>/dev/null | head -1); \
		if [ -z "$$LATEST" ]; then \
			echo "$(YELLOW)No backup files found$(NC)"; \
			exit 1; \
		fi; \
		docker compose exec -T postgres psql -U morgan -d morgan < $$LATEST; \
	else \
		docker compose exec -T postgres psql -U morgan -d morgan < $(BACKUP_FILE); \
	fi
	@echo "$(GREEN)Database restored!$(NC)"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(YELLOW)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker compose down -v postgres; \
		docker compose up -d postgres; \
		sleep 5; \
		make db-migrate; \
		echo "$(GREEN)Database reset complete!$(NC)"; \
	else \
		echo "$(BLUE)Cancelled$(NC)"; \
	fi

##@ Deployment

deploy-local: ## Deploy to local environment
	@./scripts/deploy.sh local

deploy-staging: ## Deploy to staging environment
	@./scripts/deploy.sh staging

deploy-production: ## Deploy to production environment
	@./scripts/deploy.sh production

rollback-staging: ## Rollback staging deployment
	@./scripts/deploy.sh staging --rollback

rollback-production: ## Rollback production deployment
	@./scripts/deploy.sh production --rollback

health-staging: ## Check staging environment health
	@./scripts/deploy.sh staging --health-check

health-production: ## Check production environment health
	@./scripts/deploy.sh production --health-check

##@ Cleaning

clean: ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	@echo "$(GREEN)Cleaned!$(NC)"

clean-all: clean docker-clean ## Clean everything including Docker

##@ CI/CD

ci-test: ## Run CI tests
	@./scripts/test.sh --all --coverage --parallel

ci-lint: ## Run CI linting
	@./scripts/lint.sh --all --ci

ci-security: ## Run CI security checks
	@./scripts/lint.sh --security --ci

ci: ci-lint ci-test ci-security ## Run all CI checks

##@ Information

version: ## Show current version
	@if [ -f .version ]; then \
		echo "Version: $$(cat .version)"; \
	else \
		echo "Version: $$(git describe --tags --always 2>/dev/null || echo 'unknown')"; \
	fi

info: ## Show project information
	@echo "$(BLUE)Morgan v2-0.0.1 Project Information$(NC)"
	@echo ""
	@echo "Python version: $$(python3 --version)"
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose version: $$(docker compose version 2>/dev/null || echo 'Not installed')"
	@echo "Virtual env: $$(if [ -d $(VENV) ]; then echo 'Present'; else echo 'Not found'; fi)"
	@echo ""
	@make version

status: ## Show service status
	@echo "$(BLUE)Service Status:$(NC)"
	@docker compose ps

docs: ## Open documentation
	@echo "$(BLUE)Opening documentation...$(NC)"
	@if command -v xdg-open > /dev/null; then \
		xdg-open docs/README.md; \
	elif command -v open > /dev/null; then \
		open docs/README.md; \
	else \
		echo "Please open docs/README.md manually"; \
	fi

##@ Quick Start

quickstart: install docker-build docker-up test health ## Complete setup and start
	@echo ""
	@echo "$(GREEN)=======================================$(NC)"
	@echo "$(GREEN)Morgan is ready!$(NC)"
	@echo "$(GREEN)=======================================$(NC)"
	@echo ""
	@echo "Access the services at:"
	@echo "  - Core:  http://localhost:8000"
	@echo "  - LLM:   http://localhost:8001"
	@echo "  - TTS:   http://localhost:8002"
	@echo "  - STT:   http://localhost:8003"
	@echo ""
	@echo "Next steps:"
	@echo "  make logs       - View logs"
	@echo "  make test       - Run tests"
	@echo "  make help       - See all commands"
	@echo ""
