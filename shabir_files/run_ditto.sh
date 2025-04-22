#!/bin/bash
SERVER_SCRIPT="server/server_ditto.py"
CLIENT_SCRIPT="client/client_ditto.py"

EVAL_DATASET="dataset/gbsg.csv"

NUM_ROUNDS=10

# Server address
SERVER_ADDRESS="0.0.0.0:8080"

# Client training datasets
DATASET_YOUNG="dataset/BreastCancerDataRoystonAltman_subset_A_younger.csv"
DATASET_OLD="dataset/BreastCancerDataRoystonAltman_subset_B_older.csv"

# Log file names.
SERVER_LOG="server_ditto_simple.log"
CLIENT1_LOG="client_ditto_young.log"
CLIENT2_LOG="client_ditto_old.log"

# --- Start the Server ---
echo "Starting Flower server (DITTO mode) with evaluation on ${EVAL_DATASET}..."
nohup python3 ${SERVER_SCRIPT} \
  --eval_dataset ${EVAL_DATASET} \
  --num_rounds ${NUM_ROUNDS} \
  --server_address ${SERVER_ADDRESS} \
  > ${SERVER_LOG} 2>&1 &
SERVER_PID=$!
echo "Server started with PID: ${SERVER_PID}"

sleep 5

# --- Start the Clients ---
# Client 1: uses the young
echo "Starting DITTO client for Young data (${DATASET_YOUNG})..."
nohup python3 ${CLIENT_SCRIPT} \
  --dataset_path ${DATASET_YOUNG} \
  --server_address ${SERVER_ADDRESS} \
  > ${CLIENT1_LOG} 2>&1 &
CLIENT1_PID=$!
echo "Client 1 (Young) started with PID: ${CLIENT1_PID}"

# Client 2: uses the old 
echo "Starting DITTO client for Old data (${DATASET_OLD})..."
nohup python3 ${CLIENT_SCRIPT} \
  --dataset_path ${DATASET_OLD} \
  --server_address ${SERVER_ADDRESS} \
  > ${CLIENT2_LOG} 2>&1 &
CLIENT2_PID=$!
echo "Client 2 (Old) started with PID: ${CLIENT2_PID}"

wait
