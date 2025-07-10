#!/bin/bash

# Simple temperature logger - saves raw platform-api output
# Usage: 
#   ./temp_monitor.sh [duration_seconds] [interval_seconds]
#   ./temp_monitor.sh --pid [PID] [interval_seconds]

# Parse arguments
if [ "$1" = "--pid" ]; then
    if [ -z "$2" ]; then
        echo "Error: PID required when using --pid option"
        echo "Usage: $0 --pid [PID] [interval_seconds]"
        exit 1
    fi
    
    PID_MODE=1
    TARGET_PID=$2
    INTERVAL=${3:-1}  # Default 1 second interval
    
    # Check if PID exists
    if ! kill -0 "$TARGET_PID" 2>/dev/null; then
        echo "Error: Process with PID $TARGET_PID does not exist"
        exit 1
    fi
    
    # Get process name for logging
    PROCESS_NAME=$(ps -p "$TARGET_PID" -o comm= 2>/dev/null || echo "unknown")
    
    echo "Starting temperature monitoring for PID $TARGET_PID ($PROCESS_NAME)..."
    echo "Will monitor until process exits..."
    
else
    PID_MODE=0
    DURATION=${1:-60}  # Default 60 seconds
    INTERVAL=${2:-1}   # Default 1 second interval
    
    echo "Starting temperature monitoring for ${DURATION} seconds..."
fi

LOGFILE="temp_log_$(date +%Y%m%d_%H%M%S).txt"

echo "Logging raw output to: ${LOGFILE}"
if [ $PID_MODE -eq 1 ]; then
    echo "Monitoring PID: ${TARGET_PID} (${PROCESS_NAME})"
    echo "Measuring Frequency: ${INTERVAL}s until process exits"
    echo "Monitoring PID: ${TARGET_PID} (${PROCESS_NAME})" >> "${LOGFILE}"
    echo "Measuring Frequency: ${INTERVAL}s until process exits" >> "${LOGFILE}"
else
    echo "Measuring Frequency: ${INTERVAL}s for total duration of ${DURATION}s"
    echo "Measuring Frequency: ${INTERVAL}s for total duration of ${DURATION}s" >> "${LOGFILE}"
fi

echo "Press Ctrl+C to stop early"

# Get baseline
echo "=== BASELINE ===" >> "${LOGFILE}"
echo "Timestamp: $(date)" >> "${LOGFILE}"
if [ $PID_MODE -eq 1 ]; then
    echo "Target PID: ${TARGET_PID} (${PROCESS_NAME})" >> "${LOGFILE}"
fi
sudo platform-api --print_temp c >> "${LOGFILE}"
echo "" >> "${LOGFILE}"

# Store previous output for change detection
PREV_OUTPUT=$(sudo platform-api --print_temp c)

# Start monitoring
START_TIME=$(date +%s)
if [ $PID_MODE -eq 0 ]; then
    END_TIME=$((START_TIME + DURATION))
fi

SAMPLE_COUNT=0
LOG_COUNT=1  # Already logged baseline

echo "=== MONITORING DATA ===" >> "${LOGFILE}"

# Main monitoring loop
while true; do
    # Check exit conditions
    if [ $PID_MODE -eq 1 ]; then
        # PID mode: check if process still exists
        if ! kill -0 "$TARGET_PID" 2>/dev/null; then
            echo -e "\nProcess $TARGET_PID has exited. Stopping monitoring."
            break
        fi
    else
        # Duration mode: check if time elapsed
        if [ $(date +%s) -ge $END_TIME ]; then
            break
        fi
    fi
    
    CURRENT_OUTPUT=$(sudo platform-api --print_temp c)
    SAMPLE_COUNT=$((SAMPLE_COUNT + 1))
    ELAPSED=$(($(date +%s) - START_TIME))
    
    # Check if output changed
    if [ "$CURRENT_OUTPUT" != "$PREV_OUTPUT" ]; then
        # Log timestamp and raw output
        echo "--- Sample ${LOG_COUNT} ---" >> "${LOGFILE}"
        echo "Time: $(date)" >> "${LOGFILE}"
        echo "Elapsed: ${ELAPSED}s" >> "${LOGFILE}"
        if [ $PID_MODE -eq 1 ]; then
            echo "PID Status: Running" >> "${LOGFILE}"
        fi
        echo "$CURRENT_OUTPUT" >> "${LOGFILE}"
        echo "" >> "${LOGFILE}"
        
        LOG_COUNT=$((LOG_COUNT + 1))
        PREV_OUTPUT="$CURRENT_OUTPUT"
        
        # Display with change indicator
        if [ $PID_MODE -eq 1 ]; then
            printf "\r[%3ds] PID:%d | Sample: %d | Logged: %d | CHANGED *" "$ELAPSED" "$TARGET_PID" "$SAMPLE_COUNT" "$LOG_COUNT"
        else
            printf "\r[%3ds] Sample: %d | Logged: %d | CHANGED *" "$ELAPSED" "$SAMPLE_COUNT" "$LOG_COUNT"
        fi
    else
        # Display without change indicator
        if [ $PID_MODE -eq 1 ]; then
            printf "\r[%3ds] PID:%d | Sample: %d | Logged: %d" "$ELAPSED" "$TARGET_PID" "$SAMPLE_COUNT" "$LOG_COUNT"
        else
            printf "\r[%3ds] Sample: %d | Logged: %d" "$ELAPSED" "$SAMPLE_COUNT" "$LOG_COUNT"
        fi
    fi
    
    sleep "${INTERVAL}"
done

echo -e "\nMonitoring complete!"

# Final log entry
echo "=== SUMMARY ===" >> "${LOGFILE}"
echo "End time: $(date)" >> "${LOGFILE}"
echo "Total samples: ${SAMPLE_COUNT}" >> "${LOGFILE}"
echo "Logged entries: ${LOG_COUNT}" >> "${LOGFILE}"
echo "Total duration: ${ELAPSED} seconds" >> "${LOGFILE}"

if [ $PID_MODE -eq 1 ]; then
    echo "Target PID: ${TARGET_PID} (${PROCESS_NAME})" >> "${LOGFILE}"
    if kill -0 "$TARGET_PID" 2>/dev/null; then
        echo "Process status: Still running" >> "${LOGFILE}"
    else
        echo "Process status: Exited" >> "${LOGFILE}"
    fi
else
    echo "Configured duration: ${DURATION} seconds" >> "${LOGFILE}"
fi

echo "Log saved to: ${LOGFILE}"
echo "Total samples: ${SAMPLE_COUNT}"
echo "Logged entries: ${LOG_COUNT}"

if [ $LOG_COUNT -gt 1 ]; then
    echo "Compression ratio: $((SAMPLE_COUNT / LOG_COUNT))x"
else
    echo "No temperature changes detected"
fi

if [ $PID_MODE -eq 1 ]; then
    if kill -0 "$TARGET_PID" 2>/dev/null; then
        echo "Note: Process $TARGET_PID is still running"
    else
        echo "Process $TARGET_PID has exited"
    fi
fi
