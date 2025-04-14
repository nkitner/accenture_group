#!/bin/bash

# --- Configuration ---
SERVER_SCRIPT="server/server.py"
SERVER_CONFIG="server/config.yaml"
DATASET_A="dataset/BreastCancerDataRoystonAltman_subset_A.csv"
DATASET_B="dataset/BreastCancerDataRoystonAltman_subset_B.csv"

# Output log file names.
SERVER_LOG="server.log"
CLIENT1_LOG="client_1.log"
CLIENT2_LOG="client_2.log"

# Server address to be used by clients.
SERVER_ADDRESS="0.0.0.0:8080"

# --- Start the Server ---
echo "Starting Flower server..."
nohup python3 ${SERVER_SCRIPT} --config ${SERVER_CONFIG} > ${SERVER_LOG} 2>&1 &
SERVER_PID=$!
echo "Server started with PID: ${SERVER_PID}"

# Pause to allow the server to start.
sleep 5

# --- Start the Clients ---
# Client 1 with dataset A.
echo "Starting Client 1 with ${DATASET_A}..."
nohup python3 client/client.py --dataset_path ${DATASET_A} --server_address ${SERVER_ADDRESS} > ${CLIENT1_LOG} 2>&1 &
CLIENT1_PID=$!
echo "Client 1 started with PID: ${CLIENT1_PID}"

# Client 2 with dataset B.
echo "Starting Client 2 with ${DATASET_B}..."
nohup python3 client/client.py --dataset_path ${DATASET_B} --server_address ${SERVER_ADDRESS} > ${CLIENT2_LOG} 2>&1 &
CLIENT2_PID=$!
echo "Client 2 started with PID: ${CLIENT2_PID}"

# --- Wait for Background Processes ---
wait
