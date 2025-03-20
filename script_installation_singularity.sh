wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz  # Vérifie la dernière version sur https://go.dev/dl/
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
source ~/.bashrc


sudo add-apt-repository ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer

python3 -m pip install kaleido