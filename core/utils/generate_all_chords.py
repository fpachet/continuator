from itertools import combinations

midi_min = 48
midi_max = 72

# Notes disponibles après 48
rest_notes = list(range(49, midi_max + 1))

with open("ensembles_commencant_par_48.txt", "w") as f:
    for size in range(1, 8):  # 1 à 7 notes après 48 → ensembles de taille 2 à 8
        for combo in combinations(rest_notes, size):
            ensemble = (48,) + combo
            f.write(",".join(map(str, ensemble)) + "\n")

print("Fichier généré : ensembles_commencant_par_48.txt")
