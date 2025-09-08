#!/bin/bash
prefix="172.20.83"

# des01 - des18
for i in {201..218}
do
    ip="$prefix.$i"
    
    ping -c 2 "$ip" > /dev/null
    if [ $? -eq 0 ]; then
        echo "Host $ip is reachable. Attempting SSH connection..."
        
        ssh -o "StrictHostKeyChecking=no" -o "BatchMode=yes" -o "ConnectTimeout=5" "$ip" exit
        
        if [ $? -eq 0 ]; then
            echo "Successfully connected to $ip and exited."
        else
            echo "Failed to connect to $ip or SSH timed out."
        fi
    else
        echo "Host $ip is unreachable."
    fi
done

# appl11.mass 10.40.71.42
# appl13.mass 10.40.71.125
# sanna.kask 172.20.97.101
# vinanna.kask 172.20.97.102
extra_ips=("10.40.71.42" "10.40.71.125" "172.20.97.101" "172.20.97.102")

for ip in "${extra_ips[@]}"
do
    ping -c 2 "$ip" > /dev/null
    if [ $? -eq 0 ]; then
        echo "Host $ip is reachable. Attempting SSH connection..."
        
        ssh -o "StrictHostKeyChecking=no" -o "BatchMode=yes" -o "ConnectTimeout=5" "$ip" exit
        
        if [ $? -eq 0 ]; then
            echo "Successfully connected to $ip and exited."
        else
            echo "Failed to connect to $ip or SSH timed out."
        fi
    else
        echo "Host $ip is unreachable."
    fi
done