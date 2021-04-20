import os, sys, json, tqdm
import argparse

from index_class import Index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Lean', help = 'Whether to write cache files or do everything in memory')
    parser.add_argument('--CacheDir', default=None, help = 'Where to write index cache files')
    parser.add_argument('--OutputDir', default=None, help = 'Where to write query results')
    parser.add_argument('--Recall', default=10, help='How many results to retrieve')
    parser.add_argument('--BuildIndexList', default=None)
    parser.add_argument('--QueryIndexList', default=None)
    args = parser.parse_args()

    if args.Lean and not args.CacheDir:
        print('[ERROR]: If not running in lean mode, a cache directory must be specified...')
        exit(1)

    index = Index(args.CacheDir)

    if args.BuildIndexList:
        index.train(index_image_list)
        index.add_to_index(index_image_list)

    if args.QueryIndexList:
        raw_query_output = index.query_index(query_image_list)
        print(raw_query_output)
        # json_queries = index.query_to_json(raw_query_output)
