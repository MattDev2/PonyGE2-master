import os
import random
import string

def random_string(length):
    characters = string.ascii_letters + string.digits
    random_str = ''.join(random.choice(characters) for _ in range(length))
    return random_str

input_file = 'best_genotypes.txt'
output_folder = 'alzheimer_seed_population'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(input_file, 'r') as infile:
    lines = infile.readlines()

    # Salta la prima riga
    for line in lines[1:]:
        output_file = f"{output_folder}/{random_string(8)}.txt"

        with open(output_file, 'w') as outfile:
            # Scrivi "Genotype" come prima riga del file
            outfile.write("Genotype:\n")

            # Scrivi la riga del file di input
            outfile.write(line)
