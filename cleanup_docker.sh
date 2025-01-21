#!/bin/bash

# Stop all running containers
echo "Stopping all running containers..."
docker stop $(docker ps -q) || true

# Remove all stopped containers
echo "Removing all containers..."
docker rm $(docker ps -a -q) || true

# Prune unused volumes
echo "Pruning unused volumes..."
docker volume prune -f || true

# Prune unused networks
echo "Pruning unused networks..."
docker network prune -f || true

# Remove all images
echo "Removing all images..."
docker rmi -f $(docker images -q) || true

echo "All done!"