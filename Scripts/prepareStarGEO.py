import gzip
from helper import *
import sys
import requests

import pandas as pd

r = requests.get('http://stargeo.org/api/v2/series/?limit=100000000')
assert r.ok
data = r.json()
series = pd.DataFrame(data['results'])
series.to_csv("/tmp/series.tsv", sep="\t")

#annotations = None
#url = "http://stargeo.org/api/v2/annotations/"

#while (annotations_chunk_retrieved := requests.get(url).json())["next"] != None:
#    annotations_chunk = pd.DataFrame(annotations_chunk_retrieved["results"])
#
#    if annotations is None:
#        annotations = annotations_chunk
#    else:
#        annotations = pd.concat([annotations, annotations_chunk], ignore_index=True)
#
#    url = annotations_chunk_retrieved["next"]
#    print(url)

#annotations.to_csv("/tmp/annotations.tsv", sep="\t")
#annotations = pd.read_csv("/tmp/annotations.tsv", sep="\t")

#tags = pd.read_json('http://stargeo.org/api/v2/tags/')
#tags = tags[tags.concept_name != '']
#tags.to_csv("/tmp/tags.tsv", sep="\t")

#annotations_tags = pd.merge(annotations, tags, left_on="tag_id", right_on="id", how="inner")
#annotations_tags.to_csv("/tmp/annotations_tags.tsv", sep="\t")

#print(requests.get('http://stargeo.org/api/v2/annotations/1607/samples/').text)
#print(requests.get('http://stargeo.org/api/v2/annotations/1607/').text)

#print(pd.read_json(annotations["results"]))
#data = pd.read_json("http://stargeo.org/api/v2/annotations/")
#print(data.columns)
sys.exit(1)

star_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]

# Check whether we already have a cache of summaries
if not os.path.exists(star_file_path):
    with gzip.open(all_geo_file_path) as cache_file:
        all_geo_dict = json.loads(cache_file.read())

    # Pull summaries from STARGEO
    rq = requests.get('http://stargeo.org/api/v2/series/?limit=10000000').json()

    star_geo_list = []

    for row in rq['results']:
        gse_id = row['gse_name']
        species = row['specie']

        if gse_id in all_geo_dict and species == "human":
            star_geo_list.append(gse_id)

    with gzip.open(star_file_path, 'w') as star_file:
        star_file.write(json.dumps(star_geo_list).encode())

    print(len(star_geo_list)) #24303
