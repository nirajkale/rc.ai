sudo cp display_utils/pioled_stats.service /etc/systemd/system/pioled_stats.service
sudo systemctl daemon-reload
sudo systemctl enable pioled_stats
sudo systemctl start pioled_stats