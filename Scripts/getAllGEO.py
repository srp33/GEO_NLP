from geofetch import Finder
import glob
import gzip
import joblib
import os
import requests
import sys
import time

tmp_dir_path = sys.argv[1]
out_tsv_file_path = sys.argv[2]

gse_obj = Finder()
gse_list = sorted(gse_obj.get_gse_all())

def save_gse(gse):
    try:
        tmp_file_path = f"{tmp_dir_path}/{gse}"

        if os.path.exists(tmp_file_path):
            print(f"{gse} has already been saved.")
            return False
        else:
            print(f"Retrieving metadata for {gse}.")

        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}&targ=gse&view=quick&form=text"
        gse_text = requests.get(url).text

        gse_lines = [line.rstrip("\n") for line in gse_text.strip().split("\n")]
        gse_lines = [line for line in gse_lines if line.startswith("!Series") and not (line.startswith("!Series_sample") or line.startswith("!Series_contact"))]
        gse_text = "\n".join(gse_lines)

        with open(tmp_file_path, "w") as tmp_file:
            tmp_file.write(gse_text)

        time.sleep(1)
        return True
    except:
        print(f"Error occurred for {gse}.")
        time.sleep(3)
        return False

## Last ran on January 19, 2024.
joblib.Parallel(n_jobs=8)(joblib.delayed(save_gse)(gse) for gse in gse_list)

## Sometimes the files are empty. This removes them.
#for file_path in glob.glob("Data/tmp/*"):
#    if os.path.getsize(file_path) == 0:
#        print(f"Removing {file_path} because it is empty.")
#        os.remove(file_path)

with gzip.open(out_tsv_file_path, "w") as out_tsv_file:
    header = f"GSE\tTitle\tSummary\tOverall_Design\tExperiment_Type\tGPL\tSpecies\tTaxon_ID\tSuperSeries_GSE\tSubSeries_GSEs\tPubMed_IDs\n"
    out_tsv_file.write(header.encode())

    for in_file_path in sorted(glob.glob(f"{tmp_dir_path}/*")):
        with open(in_file_path) as in_file:
            print(f"Parsing text from {in_file_path}")

            gse = os.path.basename(in_file_path)
            gse_text = in_file.read()

            gse_dict = {}
            for line in gse_text.split("\n"):
                line = line.strip()
                if len(line) == 0:
                    continue

                line_items = line.split(" = ")

                if len(line_items) < 2:
                    continue

                key = line_items[0]
                key = key.replace("!Series_", "")
                value = line_items[1]

                gse_dict.setdefault(key, []).append(value)

                subseries_gse = []
                for x in gse_dict.get("relation", []):
                    if x.startswith("SubSeries of: "):
                        subseries_gse.append(x.replace("SubSeries of: ", ""))
                gse_dict["subseries_gse"] = subseries_gse

                superseries_gse = []
                for x in gse_dict.get("relation", []):
                    if x.startswith("SuperSeries of: "):
                        superseries_gse.append(x.replace("SuperSeries of: ", ""))
                gse_dict["superseries_gse"] = superseries_gse

            for key, value_list in gse_dict.items():
                if len(value_list) == 1:
                    gse_dict[key] = value_list[0]
                else:
                    gse_dict[key] = "; ".join(value_list)

            title = gse_dict["title"]

            if title == "RETIRED": # This happened at least for GSE1829.
                continue

            summary = gse_dict.get("summary", "")
            overall_design = gse_dict.get("overall_design", "")
            experiment_type = gse_dict.get("type", "")
            gpl = gse_dict.get("platform_id", "")
            species = gse_dict.get("platform_organism", "")
            taxon_id = gse_dict.get("platform_taxid", "")
            subseries_gse = gse_dict.get("subseries_gse", "")
            superseries_gse = gse_dict.get("superseries_gse", "")
            pubmed_ids = gse_dict.get("pubmed_id", "")

            out_line = "\t".join([gse, title, summary, overall_design, experiment_type, gpl, species, taxon_id, subseries_gse, superseries_gse, pubmed_ids]) + "\n"
            out_tsv_file.write(out_line.encode())
