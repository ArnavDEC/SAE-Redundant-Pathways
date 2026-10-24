sudo apt update
sudo apt install iputils-ping npm unzip docker-buildx
sudo usermod -aG docker $USER
npm config set prefix '~/.npm-global'
npm install -g @anthropic-ai/claude-code
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc

