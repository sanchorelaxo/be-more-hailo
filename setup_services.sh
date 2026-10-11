#!/bin/bash

BMO_USER="$USER"
BMO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Setting up BMO background services..."
echo "  User: $BMO_USER"
echo "  Directory: $BMO_DIR"

# 1. Create the Hailo-Ollama Service
echo "Creating bmo-ollama.service..."
cat << EOF | sudo tee /etc/systemd/system/bmo-ollama.service
[Unit]
Description=BMO Hailo Ollama Service
After=network.target

[Service]
Type=simple
User=$BMO_USER
WorkingDirectory=$BMO_DIR
Environment="OLLAMA_HOST=0.0.0.0:8000"
ExecStart=/usr/bin/hailo-ollama serve
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# 2. Create the Web UI Service
echo "Creating bmo-web.service..."
cat << EOF | sudo tee /etc/systemd/system/bmo-web.service
[Unit]
Description=BMO Web UI Service
After=network.target bmo-ollama.service
Requires=bmo-ollama.service

[Service]
Type=simple
User=$BMO_USER
WorkingDirectory=$BMO_DIR
ExecStart=/bin/bash $BMO_DIR/start_web.sh
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 3. Create the Voice Agent Service
echo "Creating bmo-voice.service..."
BMO_UID=$(id -u "$BMO_USER")
cat << EOF | sudo tee /etc/systemd/system/bmo-voice.service
[Unit]
Description=BMO Voice Agent
After=network.target bmo-ollama.service
Requires=bmo-ollama.service

[Service]
Type=simple
User=$BMO_USER
WorkingDirectory=$BMO_DIR
Environment="DISPLAY=:0"
Environment="XAUTHORITY=/home/$BMO_USER/.Xauthority"
Environment="XDG_RUNTIME_DIR=/run/user/$BMO_UID"
Environment="PYTHONUNBUFFERED=1"
ExecStart=$BMO_DIR/venv/bin/python $BMO_DIR/agent_hailo.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 4. Fix Permissions
echo "Fixing permissions for the web app..."
sudo chown -R "$BMO_USER:$BMO_USER" "$BMO_DIR"
sudo chmod +x "$BMO_DIR"/*.sh

# 5. Reload systemd, enable, and start the services
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

echo "Enabling services to start on boot..."
sudo systemctl enable bmo-ollama.service
sudo systemctl enable bmo-web.service
sudo systemctl enable bmo-voice.service

echo "Starting services now..."
sudo systemctl start bmo-ollama.service
sudo systemctl start bmo-web.service
sudo systemctl start bmo-voice.service

echo ""
echo "========================================================"
echo "Setup Complete!"
echo "The Hailo LLM, Web UI, and Voice Agent are now running."
echo "They will automatically start whenever the Pi reboots."
echo "========================================================"
echo "To check the status of the Voice Agent: sudo systemctl status bmo-voice"
echo "To check the status of the Web UI:      sudo systemctl status bmo-web"
echo "To check the status of the LLM:         sudo systemctl status bmo-ollama"
echo "To follow the Voice Agent log:          journalctl -u bmo-voice -f"
echo "========================================================"