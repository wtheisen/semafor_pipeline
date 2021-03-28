import argparse, glob, os, subprocess
from tqdm import tqdm

def chunk_gen(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]

parser = argparse.ArgumentParser()
parser.add_argument('--RootQueryDir', help = 'Root directory containing query images', default=None)
parser.add_argument('--Email', help = 'Root directory containing query images', default=None)
parser.add_argument('--IndexSavePath', help = 'where to output query results', default=None)
parser.add_argument('--DatasetName', help = 'feature detector', default=None)
parser.add_argument('--Det', help = 'feature detector', default=None)
parser.add_argument('--Desc', help = 'feature descriptor', default=None)
parser.add_argument('--IndexImageRoot', help='nist dataset directory')
parser.add_argument('--Recall', help='output directory for the Index')
parser.add_argument('--BatchSize', help='Number of queries per batch/job')
args = parser.parse_args()

file_exts = ['png', 'jpg', 'jpeg']
img_path_list = []

for ext in tqdm(file_exts, desc='Searching for images to query...'):
    img_path_list += glob.glob(os.path.join(args.RootQueryDir, f'**/*.{ext}'), recursive = True)

query_file_dir = f'{args.IndexSavePath}/cache_{args.DatasetName}/batch_query_lists'
os.mkdir(query_file_dir)

job_num = 0
for batch in chunk_gen(img_path_list, int(args.BatchSize)):

    img_list_filename = f'{job_num}_batch_image_list.dat'
    with open(os.path.join(query_file_dir, img_list_filename), 'w+') as f:
        f.write('\n'.join(batch))

    job_file_string = f'''
#!/bin/bash
#$ -q gpu
#$ -l gpu=1
#$ -M {args.Email}
#$ -m abe

#Load conda module
module load conda
conda activate image_clustering_env

#Reset output directory (important if cache failures exist)
#rm -r ~/reddit_semafor_output/
#mkdir ~/reddit_semafor_output

#Remove output dumps from previous jobs
#rm ./semafor_job_output.out
#rm ./*.job.*

python3 provenanceFilteringDriver.py \\
        --GenericProbeList "{os.path.join(query_file_dir, img_list_filename)}" \\
        --IndexOutputDir "{args.IndexSavePath}/index_{args.DatasetName}/" \\
        --ProvenanceOutputFile "{args.IndexSavePath}/results_{args.DatasetName}/results.csv" \\
        --CacheFolder "{args.IndexSavePath}/cache_{args.DatasetName}/" \\
        --TrainedIndexParams "{args.IndexSavePath}/indextraining_{args.DatasetName}/parameters" \\
        --NISTDataset {args.IndexImageRoot} \\
        --det {args.Det} --desc {args.Desc} \\
        --outputdir {args.IndexSavePath} \\
        --Recall {args.Recall}
    '''

    job_name = f'batch_query_job_{job_num}.job'
    with open(job_name, 'a+') as job_file:
        job_file.write(job_file_string)

    job_num += 1
    print(f'Submitted job number {job_num}')
    exit(1)

    if subprocess.run(['qsub', job_name]):
        print(f'Submitted batch number {job_num} using file {job_name}')
    else:
        print(f'ERROR: Failed to submit job number {job_num}')

print(f'Submitted {job_num - 1} jobs')
