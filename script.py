import pandas as pd
import sys
import subprocess

def extractingFeaturesName(dataset):
    """Extracts feature names from a DataFrame and writes them to a text file."""
    with open("Features.txt", "w") as file:
        for col in dataset.columns:
            if col != "attack_cat":
                file.write(col + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    try:
        # Load the CSV file
        data = pd.read_csv(input_file)
        
        # Extract features and write to file
        extractingFeaturesName(data)
        print("Features extracted to Features.txt'")

        # Read each feature name from the file and run SVM.py with it
        with open("Features.txt", "r") as file:
            features = file.read().splitlines()
        
        for feature in features:
            print(f"Running SVM.py with feature: {feature}")
            subprocess.run(["python3", "SVM.py", feature])
    
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
