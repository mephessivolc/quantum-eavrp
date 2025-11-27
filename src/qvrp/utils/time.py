import time 
from datetime import timedelta

def get_time():
    return time.perf_counter() 

def run_time(start, finish):
    return timedelta(seconds=finish-start)
