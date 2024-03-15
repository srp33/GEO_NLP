import gemmapy
import gzip
import json
import sys

all_geo_json_file_path = sys.argv[1]
out_all_gemma_file_path = sys.argv[2]
out_non_gemma_file_path = sys.argv[3]

# Get all eligible GEO series
with gzip.open(all_geo_json_file_path) as all_geo_file:
    all_geo_json = json.loads(all_geo_file.read())
    all_geo_series = set([x.encode() for x in all_geo_json.keys()])

client = gemmapy.GemmaPy()

# client.api is a low-level SDK that is generated automatically
# We first obtain the total number of datasets available
total_elements = client.api.get_datasets(offset=0, limit=100).total_elements

# We obtain datasets page by page
gemma_series = set()
for i in range(0, total_elements, 100):
    if i > 0:
        print(i)

    api_response = client.api.get_datasets(offset=i, limit=100)

    for d in api_response.data:
        if d.accession is not None and d.external_database == 'GEO':
            gemma_series.add(d.accession.encode())

gemma_series_out = all_geo_series & gemma_series

with gzip.open(out_all_gemma_file_path, "w") as out_file:
    out_file.write(b"\n".join(sorted(list(gemma_series_out))))

print(f"Saved to {out_all_gemma_file_path}:")
print(len(gemma_series_out)) #5703

non_gemma_series_out = all_geo_series - gemma_series
with gzip.open(out_non_gemma_file_path, "w") as out_file:
    out_file.write(b"\n".join(sorted(list(non_gemma_series_out))))

print(f"Saved to {out_non_gemma_file_path}:")
print(len(non_gemma_series_out)) #70277
