import gemmapy
import gzip
import json
import sys

mondo_term_ids = sys.argv[1].split(",")
is_large = sys.argv[2] == "True"
all_geo_json_file_path = sys.argv[3]
out_file_path = sys.argv[4]

api_instance = gemmapy.GemmaPy()

series = set()

for mondo_term_id in mondo_term_ids:
    # Get the first page of results.
    api_response = api_instance.search_datasets([f"http://purl.obolibrary.org/obo/MONDO_{mondo_term_id}"], taxon="human", limit=100, offset=0)

    for d in api_response.data:
        if d.accession is not None and d.external_database == 'GEO':
            series.add(d.accession)

    if is_large:
        # Get the second page of results.
        api_response = api_instance.search_datasets([f"http://purl.obolibrary.org/obo/MONDO_{mondo_term_id}"], taxon="human", limit=100, offset=100)

        for d in api_response.data:
            if d.accession is not None and d.external_database == 'GEO':
                series.add(d.accession)

with gzip.open(all_geo_json_file_path) as all_file:
    all_dict = json.loads(all_file.read())

# Make sure none of the series have been excluded previously.
series = series & all_dict.keys()

series = sorted(list(series))

with open(out_file_path, "w") as out_file:
    out_file.write("\n".join(series))

print(f"Saved to {out_file_path}:")
print(len(series))
