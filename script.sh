#!/bin/bash

# Check if the input file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

input_file="$1"

# Check if the input file exists
if [ ! -f "$input_file" ]; then
    echo "File not found: $input_file"
    exit 1
fi

# Extract column names excluding 'attack_cat'
awk -F',' 'NR==1 {
    for (i=1; i<=NF; i++) {
        if ($i != "attack_cat")
            print $i
    }
}' "$input_file" > Features.txt

echo "Features extracted to Features.txt"

# Check if Features.txt was created successfully
if [ ! -f "Features.txt" ]; then
    echo "Failed to create Features.txt. Exiting."
    exit 1
fi

# Read each line from Features.txt and pass it as a parameter to SVM.py
while IFS= read -r feature
do
    echo "Running SVM.py with feature: $feature"
    python3 SVM.py "$feature"
done < "Features.txt"

