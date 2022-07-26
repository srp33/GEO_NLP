from helper import *
import sys

star_file_path = sys.argv[1]
all_geo_file_path = sys.argv[2]

# Check whether we already have a cache of summaries
if not os.path.exists(star_file_path):

    with open(all_geo_file_path) as cache_file:
        all_geo_dict = json.loads(cache_file.read())
    
    # Pull summaries from STARGEO
    rq = requests.get('http://stargeo.org/api/v2/series/?limit=10000000').json()

    star_geo_list = []

    for row in rq['results']:
        gse_id = row['gse_name']

        if gse_id in all_geo_dict:
            star_geo_list.append(gse_id)

    with open(star_file_path, 'w') as star_file:
        star_file.write(json.dumps(star_geo_list))
