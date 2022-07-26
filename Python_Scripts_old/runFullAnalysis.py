import multiprocessing
from multiprocessing import Pool

#exec(open('testPool.py').read())

pool=Pool(processes=3)

#exec(open('runSpacy.py').read())

print(f"Number of cores {multiprocessing.cpu_count()}")
