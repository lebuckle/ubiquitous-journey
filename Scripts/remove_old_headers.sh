
echo "Removing old headers"

sudo bash -c "sudo ls | grep abi | grep -v $(uname -r) | xargs -I{} mv {} ~/BACKUPS/Ubuntu_headers"
sudo apt-get autoremove 



