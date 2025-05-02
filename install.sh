#!/usr/bin/env bash

# Morgan AI Assistant Installation Script
# This script installs and sets up Morgan on a Linux system

set -e

# Text styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# Default installation directory
DEFAULT_INSTALL_DIR="/opt/morgan"

# Default config
DEFAULT_CONFIG_DIR="/opt/morgan/config"
DEFAULT_DATA_DIR="/opt/morgan/data"

# Function to display help
show_help() {
    echo -e "${BOLD}Morgan AI Assistant Installer${RESET}"
    echo
    echo "This script installs and configures Morgan, a self-hosted AI assistant for your home lab."
    echo
    echo -e "${BOLD}Usage:${RESET}"
    echo "  $0 [options]"
    echo
    echo -e "${BOLD}Options:${RESET}"
    echo "  -h, --help         Show this help message"
    echo "  -d, --directory    Installation directory (default: $DEFAULT_INSTALL_DIR)"
    echo "  --no-docker        Don't use Docker (install Python packages locally)"
    echo "  --skip-deps        Skip system dependencies installation"
    echo "  --cpu-only         Don't use GPU acceleration"
    echo "  --ha-url URL       Home Assistant URL"
    echo "  --ha-token TOKEN   Home Assistant long-lived access token"
    echo
}

# Function to print section headers
print_section() {
    echo
    echo -e "${BLUE}${BOLD}$1${RESET}"
    echo -e "${BLUE}${BOLD}$(printf '=%.0s' $(seq 1 ${#1}))${RESET}"
    echo
}

# Function to print status messages
print_status() {
    echo -e "${YELLOW}$1${RESET}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${RESET}"
}

# Function to check and install system dependencies
install_dependencies() {
    print_section "Checking System Dependencies"

    # Check for Python 3.9+
    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 --version | cut -d " " -f 2)
        python_major=$(echo $python_version | cut -d. -f1)
        python_minor=$(echo $python_version | cut -d. -f2)

        if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 9 ]); then
            print_error "Python 3.9 or higher is required (found $python_version)"
            exit 1
        else
            print_success "Python $python_version is installed"
        fi
    else
        print_error "Python 3 is not installed"

        # Try to install Python
        if command -v apt-get >/dev/null 2>&1; then
            print_status "Installing Python 3.9+ using apt..."
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv
        elif command -v dnf >/dev/null 2>&1; then
            print_status "Installing Python 3.9+ using dnf..."
            sudo dnf install -y python3 python3-pip
        else
            print_error "Could not find a package manager to install Python 3.9+"
            exit 1
        fi
    fi

    # Check for Docker if using Docker mode
    if [ "$use_docker" = true ]; then
        if command -v docker >/dev/null 2>&1 && command -v docker-compose >/dev/null 2>&1; then
            docker_version=$(docker --version | cut -d " " -f 3 | tr -d ",")
            compose_version=$(docker-compose --version | cut -d " " -f 3 | tr -d ",")
            print_success "Docker $docker_version and Docker Compose $compose_version are installed"
        else
            print_status "Installing Docker and Docker Compose..."

            if command -v apt-get >/dev/null 2>&1; then
                # Ubuntu/Debian
                sudo apt-get update
                sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

                # Add Docker's official GPG key
                curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

                # Set up the stable repository
                echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

                # Install Docker and Docker Compose
                sudo apt-get update
                sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            elif command -v dnf >/dev/null 2>&1; then
                # Fedora/CentOS
                sudo dnf -y install dnf-plugins-core
                sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
                sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            else
                print_error "Could not find a package manager to install Docker"
                exit 1
            fi

            # Start and enable Docker
            sudo systemctl start docker
            sudo systemctl enable docker

            # Add current user to the docker group
            sudo usermod -aG docker $USER
            print_status "Added user $USER to the docker group. You may need to log out and back in for this to take effect."

            print_success "Docker and Docker Compose installed"
        fi

        # Check for NVIDIA Container Toolkit if using GPU
        if [ "$use_gpu" = true ]; then
            if ! command -v nvidia-smi >/dev/null 2>&1; then
                print_error "NVIDIA drivers not found. Please install NVIDIA drivers first if you want to use GPU acceleration."
                print_status "Continuing installation without GPU support..."
                use_gpu=false
            else
                if ! grep -q "nvidia" <<< "$(docker info)"; then
                    print_status "Installing NVIDIA Container Toolkit..."

                    if command -v apt-get >/dev/null 2>&1; then
                        # Ubuntu/Debian
                        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
                        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

                        sudo apt-get update
                        sudo apt-get install -y nvidia-docker2

                        # Restart Docker
                        sudo systemctl restart docker
                    elif command -v dnf >/dev/null 2>&1; then
                        # Fedora/CentOS
                        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo

                        sudo dnf install -y nvidia-docker2

                        # Restart Docker
                        sudo systemctl restart docker
                    else
                        print_error "Could not find a package manager to install NVIDIA Container Toolkit"
                        print_status "Continuing installation without GPU support..."
                        use_gpu=false
                    fi
                fi
            fi
        fi
    else
        # Install Python development packages and other dependencies
        if command -v apt-get >/dev/null 2>&1; then
            print_status "Installing Python development packages using apt..."
            sudo apt-get update
            sudo apt-get install -y python3-dev build-essential libffi-dev
        elif command -v dnf >/dev/null 2>&1; then
            print_status "Installing Python development packages using dnf..."
            sudo dnf install -y python3-devel gcc gcc-c++ libffi-devel
        else
            print_error "Could not find a package manager to install Python development packages"
            print_status "You may need to install these manually: python3-dev, build-essential, libffi-dev"
        fi
    fi
}

# Function to create directory structure
create_directory_structure() {
    print_section "Creating Directory Structure"

    print_status "Creating installation directory: $install_dir"
    mkdir -p "$install_dir"

    # Create subdirectories
    for dir in core llm tts stt home-assistant web-ui; do
        mkdir -p "$install_dir/$dir"
        print_status "Created $install_dir/$dir"
    done

    # Create data subdirectories
    mkdir -p "$install_dir/data/models/llm"
    mkdir -p "$install_dir/data/models/tts"
    mkdir -p "$install_dir/data/models/stt"
    mkdir -p "$install_dir/data/voices"
    mkdir -p "$install_dir/data/conversations"
    mkdir -p "$install_dir/data/logs"
    print_status "Created data directories"

    # Create config directory
    mkdir -p "$install_dir/config"
    print_status "Created config directory"

    # Create scripts directory
    mkdir -p "$install_dir/scripts"
    print_status "Created scripts directory"

    print_success "Directory structure created"
}

# Function to clone the repository or copy files
setup_files() {
    print_section "Setting Up Files"

    # Check if repository exists
    if command -v git >/dev/null 2>&1; then
        print_status "Cloning repository..."

        if [ ! -d "$install_dir/.git" ]; then
            # Clone the repository
            git clone https://github.com/yourusername/morgan-assistant.git "$install_dir/temp"

            # Move files to installation directory
            cp -r "$install_dir/temp"/* "$install_dir/"
            cp -r "$install_dir/temp/.git" "$install_dir/"

            # Remove temporary directory
            rm -rf "$install_dir/temp"

            print_success "Repository cloned to $install_dir"
        else
            # Repository already exists, update it
            pushd "$install_dir" > /dev/null
            git pull
            popd > /dev/null

            print_success "Repository updated"
        fi
    else
        print_status "Git not found, copying files instead..."

        # Copy current script directory files
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        cp -r "$script_dir"/* "$install_dir/"

        print_success "Files copied to $install_dir"
    fi

    # Set permissions
    chmod +x "$install_dir/scripts"/*.sh 2>/dev/null || true
    chmod +x "$install_dir/core/app.py" 2>/dev/null || true
}

# Function to create configuration files
create_config_files() {
    print_section "Creating Configuration Files"

    if [ ! -f "$install_dir/config/core.yaml" ]; then
        print_status "Creating core.yaml..."
        cat > "$install_dir/config/core.yaml" << EOF
system:
  name: Morgan
  log_level: info
  data_dir: $DEFAULT_DATA_DIR
  max_history: 20
  context_timeout: 1800
  save_interval: 60

services:
  llm:
    url: http://llm-service:8001
    model: mistral
    system_prompt: "You are Morgan, a helpful and friendly home assistant AI. You assist with smart home controls, answer questions, and perform various tasks."
    max_tokens: 1000
    temperature: 0.7

  tts:
    url: http://tts-service:8002
    default_voice: morgan_default

  stt:
    url: http://stt-service:8003
    model: whisper-large-v3

home_assistant:
  url: $ha_url
  token: $ha_token
  reconnect_interval: 10

api:
  host: 0.0.0.0
  port: 8000
  cors_origins:
    - "*"
  auth_enabled: false
EOF
        print_success "Created core.yaml"
    else
        print_status "core.yaml already exists, skipping"
    fi

    if [ ! -f "$install_dir/config/handlers.yaml" ]; then
        print_status "Creating handlers.yaml..."
        cat > "$install_dir/config/handlers.yaml" << EOF
handlers:
  home_assistant:
    enabled: true
    domains:
      - light
      - switch
      - climate
      - media_player

  information:
    enabled: true
    weather_api_key: ""

  system:
    enabled: true
    allow_restart: true
    allow_update: true
EOF
        print_success "Created handlers.yaml"
    else
        print_status "handlers.yaml already exists, skipping"
    fi

    if [ ! -f "$install_dir/config/devices.yaml" ]; then
        print_status "Creating devices.yaml..."
        cat > "$install_dir/config/devices.yaml" << EOF
device_groups:
  living_room:
    - light.living_room
    - media_player.living_room_tv
    - climate.living_room
  kitchen:
    - light.kitchen
    - switch.coffee_maker

device_aliases:
  tv: media_player.living_room_tv
  main_lights: light.living_room
  coffee: switch.coffee_maker
EOF
        print_success "Created devices.yaml"
    else
        print_status "devices.yaml already exists, skipping"
    fi

    if [ ! -f "$install_dir/config/voices.yaml" ]; then
        print_status "Creating voices.yaml..."
        cat > "$install_dir/config/voices.yaml" << EOF
voices:
  morgan_default:
    description: "Default Morgan voice"
    type: "preset"
    preset_id: 12
EOF
        print_success "Created voices.yaml"
    else
        print_status "voices.yaml already exists, skipping"
    fi

    print_success "Configuration files created"
}

# Function to create Docker Compose file
create_docker_compose() {
    print_section "Creating Docker Compose File"

    if [ "$use_docker" = true ]; then
        print_status "Creating docker-compose.yml..."

        # Create Docker Compose file
        cat > "$install_dir/docker-compose.yml" << EOF
version: '3.8'

services:
  core:
    build:
      context: ./core
      dockerfile: Dockerfile
    container_name: morgan-core
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./config:/opt/morgan/config
      - ./data:/opt/morgan/data
    depends_on:
      - llm-service
      - tts-service
      - stt-service
    networks:
      - morgan-net

  llm-service:
    image: morgan/llm-service:latest
    container_name: morgan-llm
    restart: unless-stopped
    ports:
      - "8001:8001"
    volumes:
      - ./data/models/llm:/models
EOF

        # Add GPU configuration if using GPU
        if [ "$use_gpu" = true ]; then
            cat >> "$install_dir/docker-compose.yml" << EOF
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
        fi

        # Continue Docker Compose file
        cat >> "$install_dir/docker-compose.yml" << EOF
    networks:
      - morgan-net

  tts-service:
    image: morgan/tts-service:latest
    container_name: morgan-tts
    restart: unless-stopped
    ports:
      - "8002:8002"
    volumes:
      - ./data/models/tts:/models
      - ./data/voices:/voices
EOF

        # Add GPU configuration if using GPU
        if [ "$use_gpu" = true ]; then
            cat >> "$install_dir/docker-compose.yml" << EOF
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
        fi

        # Continue Docker Compose file
        cat >> "$install_dir/docker-compose.yml" << EOF
    networks:
      - morgan-net

  stt-service:
    image: morgan/stt-service:latest
    container_name: morgan-stt
    restart: unless-stopped
    ports:
      - "8003:8003"
    volumes:
      - ./data/models/stt:/models
EOF

        # Add GPU configuration if using GPU
        if [ "$use_gpu" = true ]; then
            cat >> "$install_dir/docker-compose.yml" << EOF
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
        fi

        # Complete Docker Compose file
        cat >> "$install_dir/docker-compose.yml" << EOF
    networks:
      - morgan-net

  web-ui:
    image: morgan/web-ui:latest
    container_name: morgan-web
    restart: unless-stopped
    ports:
      - "80:80"
    volumes:
      - ./config/web:/config
    depends_on:
      - core
    networks:
      - morgan-net

networks:
  morgan-net:
    driver: bridge
EOF

        print_success "Created docker-compose.yml"
    else
        print_status "Skipping Docker Compose file creation (non-Docker mode)"
    fi
}

# Function to create Python virtual environment and install packages
setup_python_env() {
    print_section "Setting Up Python Environment"

    if [ "$use_docker" = false ]; then
        print_status "Creating Python virtual environment..."

        # Create virtual environment
        python3 -m venv "$install_dir/venv"

        # Activate virtual environment
        source "$install_dir/venv/bin/activate"

        # Install required packages
        print_status "Installing required Python packages..."
        pip install --upgrade pip
        pip install -r "$install_dir/core/requirements.txt"

        # Create activation script
        cat > "$install_dir/activate.sh" << EOF
#!/bin/bash
source "$install_dir/venv/bin/activate"
export MORGAN_CONFIG_DIR="$install_dir/config"
export MORGAN_DATA_DIR="$install_dir/data"
echo "Morgan environment activated"
EOF
        chmod +x "$install_dir/activate.sh"

        print_success "Python environment set up"
    else
        print_status "Skipping Python environment setup (Docker mode)"
    fi
}

# Function to create service files
create_service_files() {
    print_section "Creating Service Files"

    if [ "$use_docker" = true ]; then
        print_status "Creating Docker service file..."

        # Create systemd service file for Docker mode
        sudo bash -c "cat > /etc/systemd/system/morgan.service" << EOF
[Unit]
Description=Morgan AI Assistant
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=$install_dir
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF
    else
        print_status "Creating Python service file..."

        # Create systemd service file for Python mode
        sudo bash -c "cat > /etc/systemd/system/morgan.service" << EOF
[Unit]
Description=Morgan AI Assistant
After=network.target

[Service]
Type=simple
WorkingDirectory=$install_dir/core
ExecStart=$install_dir/venv/bin/python app.py
Environment="MORGAN_CONFIG_DIR=$install_dir/config"
Environment="MORGAN_DATA_DIR=$install_dir/data"
Restart=always
User=$USER
Group=$USER

[Install]
WantedBy=multi-user.target
EOF
    fi

    # Reload systemd
    sudo systemctl daemon-reload

    print_success "Service files created"
}

# Function to download models
download_models() {
    print_section "Downloading Models"

    print_status "This step requires an internet connection and may take some time..."

    # Create the download script
    cat > "$install_dir/scripts/download_models.sh" << EOF
#!/bin/bash

# Models directory
MODELS_DIR="$install_dir/data/models"

# Create directories
mkdir -p "\$MODELS_DIR/llm"
mkdir -p "\$MODELS_DIR/tts"
mkdir -p "\$MODELS_DIR/stt"

# Download LLM model (Mistral)
echo "Downloading LLM model (Mistral)..."
if [ ! -f "\$MODELS_DIR/llm/mistral-7b-v0.1.Q4_0.gguf" ]; then
    curl -L -o "\$MODELS_DIR/llm/mistral-7b-v0.1.Q4_0.gguf" https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf
else
    echo "LLM model already exists, skipping"
fi

# Download TTS model (Piper)
echo "Downloading TTS model (Piper)..."
if [ ! -f "\$MODELS_DIR/tts/en_US-lessac-medium.onnx" ]; then
    curl -L -o "\$MODELS_DIR/tts/en_US-lessac-medium.onnx" https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
else
    echo "TTS model already exists, skipping"
fi

# Download STT model (Whisper)
echo "Downloading STT model (Whisper)..."
if [ ! -f "\$MODELS_DIR/stt/whisper-tiny.en.pt" ]; then
    curl -L -o "\$MODELS_DIR/stt/whisper-tiny.en.pt" https://huggingface.co/datasets/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin
else
    echo "STT model already exists, skipping"
fi

echo "Model downloads complete"
EOF

    # Make the script executable
    chmod +x "$install_dir/scripts/download_models.sh"

    # Run the download script
    "$install_dir/scripts/download_models.sh"

    print_success "Models downloaded"
}

# Function to complete the installation
complete_installation() {
    print_section "Installation Complete"

    echo -e "${GREEN}${BOLD}Morgan AI Assistant has been installed successfully!${RESET}"
    echo
    echo -e "Installation directory: ${BOLD}$install_dir${RESET}"
    echo

    if [ "$use_docker" = true ]; then
        echo -e "${BOLD}To start Morgan:${RESET}"
        echo "  cd $install_dir"
        echo "  docker-compose up -d"
        echo
        echo -e "${BOLD}To stop Morgan:${RESET}"
        echo "  cd $install_dir"
        echo "  docker-compose down"
    else
        echo -e "${BOLD}To activate the Morgan environment:${RESET}"
        echo "  source $install_dir/activate.sh"
        echo
        echo -e "${BOLD}To start Morgan:${RESET}"
        echo "  cd $install_dir/core"
        echo "  python app.py"
    fi

    echo
    echo -e "${BOLD}To enable automatic startup:${RESET}"
    echo "  sudo systemctl enable morgan"
    echo "  sudo systemctl start morgan"
    echo
    echo -e "${BOLD}Web Interface:${RESET}"
    echo "  http://localhost:8000/ui"
    echo
    echo -e "${BOLD}API Endpoint:${RESET}"
    echo "  http://localhost:8000/api"
    echo
    echo -e "${YELLOW}${BOLD}Note:${RESET} You may need to log out and back in for Docker permissions to take effect."
    echo
}

# Parse command line arguments
install_dir="$DEFAULT_INSTALL_DIR"
use_docker=true
skip_deps=false
use_gpu=true
ha_url="http://homeassistant:8123"
ha_token="your_long_lived_access_token"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -d|--directory)
            install_dir="$2"
            shift 2
            ;;
        --no-docker)
            use_docker=false
            shift
            ;;
        --skip-deps)
            skip_deps=true
            shift
            ;;
        --cpu-only)
            use_gpu=false
            shift
            ;;
        --ha-url)
            ha_url="$2"
            shift 2
            ;;
        --ha-token)
            ha_token="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run installation steps
echo -e "${BOLD}Morgan AI Assistant Installer${RESET}"
echo
echo -e "Installation directory: ${BOLD}$install_dir${RESET}"
echo -e "Use Docker: ${BOLD}$use_docker${RESET}"
echo -e "Use GPU: ${BOLD}$use_gpu${RESET}"
echo -e "Home Assistant URL: ${BOLD}$ha_url${RESET}"
echo

# Confirm installation
read -p "Do you want to continue with the installation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled"
    exit 0
fi

# Run installation steps
if [ "$skip_deps" = false ]; then
    install_dependencies
fi

create_directory_structure
setup_files
create_config_files
create_docker_compose
setup_python_env
create_service_files
download_models
complete_installation