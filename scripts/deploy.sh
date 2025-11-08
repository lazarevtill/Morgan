#!/usr/bin/env bash
#
# Morgan v2-0.0.1 - Deployment Script
#
# This script handles deployment to different environments.
# It is idempotent and includes rollback capabilities.
#
# Usage:
#   ./scripts/deploy.sh [ENVIRONMENT] [OPTIONS]
#
# Environments:
#   local              Deploy locally (default)
#   staging            Deploy to staging server
#   production         Deploy to production server
#
# Options:
#   --version VERSION  Deploy specific version/tag
#   --backup           Create backup before deployment
#   --no-backup        Skip backup (not recommended for production)
#   --rollback         Rollback to previous version
#   --health-check     Perform health checks only
#   --dry-run          Show what would be deployed
#   --force            Force deployment without confirmations
#   --help             Show this help message
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
ENVIRONMENT="local"
VERSION=""
BACKUP=true
ROLLBACK=false
HEALTH_CHECK_ONLY=false
DRY_RUN=false
FORCE=false

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${MAGENTA}[STEP]${NC} $1"
}

# Show help
show_help() {
    grep '^#' "$0" | grep -v '#!/usr/bin/env' | sed 's/^# //g' | sed 's/^#//g'
    exit 0
}

# Parse arguments
parse_args() {
    # First argument is environment if it doesn't start with --
    if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
        ENVIRONMENT="$1"
        shift
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                VERSION="$2"
                shift 2
                ;;
            --backup)
                BACKUP=true
                shift
                ;;
            --no-backup)
                BACKUP=false
                shift
                ;;
            --rollback)
                ROLLBACK=true
                shift
                ;;
            --health-check)
                HEALTH_CHECK_ONLY=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

# Confirm action
confirm() {
    if [[ "$FORCE" == true ]]; then
        return 0
    fi

    local message="$1"
    echo -e "${YELLOW}[CONFIRM]${NC} $message"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deployment cancelled"
        exit 0
    fi
}

# Get current version
get_current_version() {
    if [[ -f "$PROJECT_ROOT/.version" ]]; then
        cat "$PROJECT_ROOT/.version"
    else
        git describe --tags --always 2>/dev/null || echo "unknown"
    fi
}

# Health check function
health_check() {
    local base_url="$1"
    local service_name="$2"
    local max_attempts="${3:-10}"

    log_info "Checking health of $service_name..."

    for i in $(seq 1 $max_attempts); do
        if curl -f -s "$base_url/health" >/dev/null 2>&1; then
            log_success "$service_name is healthy"
            return 0
        fi
        log_info "Attempt $i/$max_attempts failed, retrying in 5s..."
        sleep 5
    done

    log_error "$service_name health check failed after $max_attempts attempts"
    return 1
}

# Check all services health
check_all_services() {
    local base_url="$1"
    local failed=0

    log_step "Running health checks..."

    health_check "$base_url:8000" "Core Service" || failed=$((failed + 1))
    health_check "$base_url:8001" "LLM Service" || failed=$((failed + 1))
    health_check "$base_url:8002" "TTS Service" || failed=$((failed + 1))
    health_check "$base_url:8003" "STT Service" || failed=$((failed + 1))

    if [[ $failed -eq 0 ]]; then
        log_success "All services are healthy!"
        return 0
    else
        log_error "$failed services failed health checks"
        return 1
    fi
}

# Create backup
create_backup() {
    if [[ "$BACKUP" == false ]]; then
        log_info "Skipping backup (--no-backup specified)"
        return 0
    fi

    log_step "Creating backup..."

    local backup_dir="$PROJECT_ROOT/backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="morgan_backup_${timestamp}"

    mkdir -p "$backup_dir"

    if [[ "$ENVIRONMENT" == "local" ]]; then
        # Backup local database
        if docker compose ps postgres | grep -q "Up"; then
            log_info "Backing up PostgreSQL database..."
            docker compose exec -T postgres pg_dump -U morgan morgan > "$backup_dir/${backup_name}.sql"
            log_success "Database backup created: $backup_dir/${backup_name}.sql"
        else
            log_warning "PostgreSQL not running, skipping database backup"
        fi
    else
        log_info "Remote backup should be handled by deployment system"
    fi

    # Save current version info
    get_current_version > "$backup_dir/${backup_name}.version"

    log_success "Backup completed: $backup_name"
}

# Deploy locally
deploy_local() {
    log_step "Deploying to local environment..."

    cd "$PROJECT_ROOT"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would execute the following:"
        log_info "  1. Pull latest code"
        log_info "  2. Build Docker images"
        log_info "  3. Start services"
        return 0
    fi

    # Create backup
    create_backup

    # Build and start services
    log_info "Building Docker images..."
    docker compose build --pull

    log_info "Starting services..."
    docker compose up -d

    # Wait for services to be ready
    sleep 10

    # Health checks
    if check_all_services "http://localhost"; then
        log_success "Local deployment completed successfully!"

        # Save version
        get_current_version > .version

        # Show service URLs
        echo ""
        log_info "Services available at:"
        echo "  - Core:  http://localhost:8000"
        echo "  - LLM:   http://localhost:8001"
        echo "  - TTS:   http://localhost:8002"
        echo "  - STT:   http://localhost:8003"
        echo ""
    else
        log_error "Deployment completed but some services failed health checks"
        return 1
    fi
}

# Deploy to staging
deploy_staging() {
    log_step "Deploying to staging environment..."

    confirm "This will deploy to STAGING environment"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would deploy to staging server"
        return 0
    fi

    local staging_host="${STAGING_HOST:-staging.lazarev.cloud}"
    local staging_user="${STAGING_USER:-deploy}"

    log_info "Deploying to $staging_host..."

    # Deploy via SSH
    ssh "$staging_user@$staging_host" << EOF
        set -e
        cd /opt/morgan
        git pull origin staging
        docker compose pull
        docker compose up -d --remove-orphans
        docker system prune -af
EOF

    sleep 30

    # Remote health check
    if check_all_services "https://staging-morgan.lazarev.cloud"; then
        log_success "Staging deployment completed successfully!"
    else
        log_error "Staging deployment failed health checks"
        return 1
    fi
}

# Deploy to production
deploy_production() {
    log_step "Deploying to production environment..."

    confirm "⚠️  This will deploy to PRODUCTION environment! ⚠️"

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would deploy to production server"
        return 0
    fi

    local prod_host="${PRODUCTION_HOST:-morgan.lazarev.cloud}"
    local prod_user="${PRODUCTION_USER:-deploy}"

    log_warning "Deploying to PRODUCTION: $prod_host"

    # Create remote backup
    log_info "Creating production backup..."
    ssh "$prod_user@$prod_host" << EOF
        set -e
        cd /opt/morgan
        mkdir -p backups
        docker compose exec -T postgres pg_dump -U morgan morgan > backups/backup_\$(date +%Y%m%d_%H%M%S).sql
EOF

    # Deploy
    log_info "Deploying to production..."
    ssh "$prod_user@$prod_host" << EOF
        set -e
        cd /opt/morgan
        git fetch --tags
        ${VERSION:+git checkout $VERSION}
        docker compose pull
        docker compose up -d --remove-orphans
        docker system prune -af
EOF

    sleep 60

    # Health checks with retries
    if check_all_services "https://$prod_host" 20; then
        log_success "Production deployment completed successfully!"

        # Create deployment marker
        ssh "$prod_user@$prod_host" << EOF
            echo "\$(date +%Y-%m-%d_%H:%M:%S) - Deployed version: ${VERSION:-latest}" >> /opt/morgan/deployments.log
EOF

        log_info "Deployment logged on production server"
    else
        log_error "Production deployment failed health checks!"
        log_error "Consider rolling back with: $0 production --rollback"
        return 1
    fi
}

# Rollback deployment
rollback_deployment() {
    log_step "Rolling back deployment..."

    confirm "⚠️  This will rollback the deployment! ⚠️"

    case "$ENVIRONMENT" in
        local)
            log_info "Rolling back local deployment..."

            cd "$PROJECT_ROOT"

            # Stop services
            docker compose down

            # Find latest backup
            local latest_backup=$(ls -t backups/*.sql 2>/dev/null | head -1)

            if [[ -n "$latest_backup" ]]; then
                log_info "Restoring from backup: $latest_backup"
                docker compose up -d postgres
                sleep 5
                docker compose exec -T postgres psql -U morgan -d morgan < "$latest_backup"
                log_success "Database restored"
            else
                log_warning "No backup found"
            fi

            # Restart services
            docker compose up -d

            log_success "Local rollback completed"
            ;;

        staging|production)
            local host="$ENVIRONMENT.lazarev.cloud"
            local user="deploy"

            log_info "Rolling back on $host..."

            ssh "$user@$host" << EOF
                set -e
                cd /opt/morgan
                docker compose down
                git reset --hard HEAD~1
                docker compose up -d
EOF

            log_success "Rollback completed on $host"
            ;;
    esac
}

# Main function
main() {
    echo ""
    log_info "======================================="
    log_info "Morgan v2-0.0.1 Deployment Script"
    log_info "======================================="
    echo ""

    parse_args "$@"

    # Show current version
    local current_version=$(get_current_version)
    log_info "Current version: $current_version"
    if [[ -n "$VERSION" ]]; then
        log_info "Target version: $VERSION"
    fi
    echo ""

    # Health check only mode
    if [[ "$HEALTH_CHECK_ONLY" == true ]]; then
        case "$ENVIRONMENT" in
            local)
                check_all_services "http://localhost"
                ;;
            staging)
                check_all_services "https://staging-morgan.lazarev.cloud"
                ;;
            production)
                check_all_services "https://morgan.lazarev.cloud"
                ;;
        esac
        exit $?
    fi

    # Rollback mode
    if [[ "$ROLLBACK" == true ]]; then
        rollback_deployment
        exit $?
    fi

    # Normal deployment
    case "$ENVIRONMENT" in
        local)
            deploy_local
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            log_info "Valid environments: local, staging, production"
            exit 1
            ;;
    esac

    echo ""
    log_success "======================================="
    log_success "Deployment completed successfully!"
    log_success "======================================="
    echo ""
}

# Run main function
main "$@"
