#!/bin/bash
#  
#  To debug the Model Loading and Execution on NPU using interrupts
#  
#
LOG_FILE="npu_interrupt_log.csv"
echo "Timestamp,Interrupt_Count" > $LOG_FILE

while true; do
    timestamp=$(date +%s.%N)
    count=$(grep 'galcore:3d' /proc/interrupts | awk '{print $2}')
    echo "$timestamp,$count" >> $LOG_FILE
    usleep 10000  # 10 ms
done

