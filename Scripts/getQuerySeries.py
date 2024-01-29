import gemmapy
import sys

mondo_term_ids = sys.argv[1].split(",")
is_large = sys.argv[2] == "True"
out_file_path = sys.argv[3]

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

series = sorted(list(series))

with open(out_file_path, "w") as out_file:
    out_file.write("\n".join(series))

print(f"Saved to {out_file_path}:")
print(len(series))
