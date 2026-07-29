import os
import glob
from datetime import datetime

def rreplace(s, old, new, n):
    """Replace the n rightmost occurrences of a string in the given string"""
    li = s.rsplit(old, n)
    return new.join(li)

def build_dataset_index(base_path):
    # Subdirectories im Base Path finden und prüfen, ob sie .yaml-Files enthalten
    # Außerdem prüfen, ob es zu jedem .yaml-GameState eine gleichnamige .txt-Datei mit den zugehörigen Win Percentages gibt 
    non_empty_subdirs = []
    for subdir in next(os.walk(base_path))[1]:
        yaml_files = glob.glob(os.path.join(base_path, subdir, "input/*.yaml"))
        txt_files = glob.glob(os.path.join(base_path, subdir, "output/*.txt"))
        if len(yaml_files) > 0:
            non_empty_subdirs.append(subdir)
        print(f"Subdir \"{subdir}\" contains {len(yaml_files)} YAML files in \"input\" directory and {len(txt_files)} TXT files in \"output\" directory.")

        # Optimierung für schnelleres Cross-Checking bei gleicher Reihenfolge
        yaml_files_set = set(yaml_files)
        txt_files_set = set(txt_files)

        # Prüfen, ob sich die YAML- und TXT-Files 1:1 zueinander zuordnen lassen
        # Erst nach fehlenden TXT-Files suchen
        for yaml_file in yaml_files:
            # .yaml Extension durch .txt ersetzen
            corresponding_file = rreplace(yaml_file, ".yaml", ".txt", 1)
            # "input"-Ordner durch "output"-Ordner ersetzen
            corresponding_file = rreplace(corresponding_file, "input", "output", 1)
            # Schauen ob die Datei in txt_files existiert
            if corresponding_file not in txt_files_set:
                print(f"Missing TXT file: \"{corresponding_file}\"")
        # Dann nach fehlenden YAML-Files suchen
        for txt_file in txt_files:
            # .txt Extension durch .yaml ersetzen
            corresponding_file = rreplace(txt_file, ".txt", ".yaml", 1)
            # "output"-Ordner durch "input"-Ordner ersetzen
            corresponding_file = rreplace(corresponding_file, "output", "input", 1)
            # Schauen ob die Datei in yaml_files existiert
            if corresponding_file not in yaml_files_set:
                print(f"Missing YAML file: \"{corresponding_file}\"")

    
    input("Continue building index on non-empty subdirs?")

    # Index aus nicht-leeren Subdirectories aufbauen
    index = []

    next_group_idx = -1
    next_idx = 0

    for subdir in non_empty_subdirs:
        subdir_name = os.path.basename(subdir)

        # Dateien in Subdir auflisten
        files = glob.glob(os.path.join(base_path, subdir, "input/*.yaml"))

        # Temporäre Liste aller Dateien im Subdirectory zum Sortieren und Gruppieren
        files_in_subdir = []

        for file in files:
            # Dateipfad und Dateinamen auslesen
            full_path = os.path.abspath(file)
            basename = os.path.splitext(os.path.basename(file))[0]

            # Dateinamen in Run-Index und State-Index unterteilen (z.B. "r919_s4" => 919, 4)
            run_idx_in_subdir, state_idx_in_run = map(int, basename.replace("r", "").replace("s", "").split("_"))

            #print(run_idx_in_subdir, state_idx_in_run, basename, full_path)
            #print(file, os.path.abspath(file), basename, os.path.basename(os.path.dirname(os.path.dirname(file))))

            files_in_subdir.append([full_path, run_idx_in_subdir, state_idx_in_run])

        # Dateinamen sortieren
        files_in_subdir.sort(key=lambda entry: (entry[1], entry[2]))

        #print(*files_in_subdir, sep="\n")

        last_seen_run = -1
        # Dateinamen gruppieren und indizieren
        for entry in files_in_subdir:
            full_path, run_idx_in_subdir, state_idx_in_run = entry
            group_idx = None
            entry_idx = None

            # Schauen ob wir noch an der gleichen Gruppe sitzen, sonst neuen Index zuweisen
            if run_idx_in_subdir != last_seen_run:
                next_group_idx += 1
                last_seen_run = run_idx_in_subdir
            group_idx = next_group_idx

            # Entry neuen Index geben
            entry_idx = next_idx
            next_idx += 1
            
            entry_name = os.path.splitext(os.path.basename(full_path))[0]

            # Zu Index hinzufügen (Idx, GroupIdx, SubDir, EntryName)
            index.append([entry_idx, group_idx, subdir_name, entry_name])
    
    # Index als Datei speichern
    output_file_name = "dataset_index_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    with open(output_file_name, "w") as output_file:
        output_file.write("entry_idx,group_idx,subdir_name,entry_name\n")
        for entry in index:
            entry_idx, group_idx, subdir_name, entry_name = entry
            output_file.write(f"{entry_idx},{group_idx},{subdir_name},{entry_name}\n")
        print(f"Saved index as \"{os.path.realpath(output_file.name)}\"!")


if __name__ == "__main__":
    base_path = input("Enter base path for dataset index scanning (Where the immediate subdirectories EACH contain an 'input' directory for YAML game states and an 'output' directory for TXT win rates). Default: Current\n")
    if base_path.strip() == "":
        base_path = "."
    input(f"Processing path \"{os.path.abspath(base_path)}\" for indexing. Press [Enter] to continue.")
    build_dataset_index(base_path)
