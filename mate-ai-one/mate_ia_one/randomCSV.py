import csv
import random


def shuffle_csv(input_file, output_file, has_header=False):
    # Legge il file CSV di input
    with open(input_file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

    # Gestisce l'header se specificato
    if has_header and len(rows) > 0:
        header = rows[0]  # Salva la prima riga come header
        data = rows[1:]  # Prende le righe rimanenti
        random.shuffle(data)  # Mescola solo i dati
        shuffled_rows = [header] + data  # Ricombina header con dati mescolati
    else:
        random.shuffle(rows)  # Mescola tutte le righe
        shuffled_rows = rows

    # Scrive il file CSV di output
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(shuffled_rows)

shuffle_csv("./dataset/tactic_evals.csv", "dataset/shuffledTaticEvals.csv")