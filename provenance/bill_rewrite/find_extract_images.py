import os, sys, json, tqdm
import argparse

finished = False
file_list = []

def find_images(directories, dataset_path_name):
    fname_to_path_dict = {}
    fname_to_ID_dict = {}
    ID_to_name_dict = {}
    fcount = 0

    try:
        for d in directories:
            print("searching " + d)

            for root, dirs, files in tqdm(os.walk(d, followlinks=True)):
                num_length = len(dirs)

                for f in files:
                    file_ID = f.split('.')[:-1]

                    if file_ID not in fname_to_path_dict:
                        fname_path = os.path.join(os.path.abspath(root), f)
                        fname_to_path_dict[file_ID] = fname_path
                        file_list.append(fname_path)
                        fcount += 1
                    else:
                        print('[WARNING]: file already detected, skip...')

            name_to_path_dict["_rootDirectory_"] = os.path.abspath(directory_to_index)
    except Exception as e:
        print(e)

    finished = True
    output_path = os.path.join(dataset_path_name + "_pathmap.json")
    output_list_path = os.path.join(dataset_path_name + "_filelist.txt")

    with open(output_path, 'w') as f:
        json.dump(name_to_path_dict, f)
    with open(output_list_path, 'w') as fp:
        fp.write('\n'.join(file_list))

    return name_to_path_dict, file_list

def process_images(file_list, consumer = False)
    with concurrent.futures.ProcessPoolExecutor(max_workers = args.PE) as executor:
        if not consumer:
            while len(file_list) > 0:
                executor.submit(image_path, file_list.pop())
        else:
            while not finished:
                executor.submit(image_path, file_list.pop())

def extract_image_features(image_path):
    image_path = image_path.rstrip()

    try:
        filename = os.path.basename(image_path);
        feature_dict  = featureExtractor.processImage(image_path, tfcores = args.TFCores)

        outpath = os.path.join(args.OutputDir, f'features_{args.DatasetName}', featureDict['supplemental_information']['value'].key)
        with open(outpath,'wb') as of:
            of.write(featureDict['supplemental_information']['value']._data)

    except Exception as e:
        print(f"[WARNING]: Problem with {file_path}\n\t[EXCEPTION]: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ImageDirectoryList', help = 'Comma separated list of image directories')
    parser.add_argument('--OutputDir', help = 'Path to output directory')
    parser.add_argument('--DatasetName', help = 'Name of the output dataset')
    parser.add_argument('--Parallel', type = int, default = 0)
    parser.add_argument('--FindOnly', type = int, default = 0)
    parser.add_argument('--InputFileList', default=None)
    parser.add_argument('--PE', type=int, default=1)
    parser.add_argument('--TFCores', type=int, default=1)
    parser.add_argument('--det', help='Keypoint Detection method', default='SURF')
    parser.add_argument('--desc', help='Feature Description method', default='SURF')
    parser.add_argument('--kmax', type=int, default=5000, help='Max keypoints per image')
    args = parser.parse_args()

    numJobs = 10
    featureExtractor = featureExtraction(args.det, args.desc, args.kmax)

    if args.Parallel and not args.FindOnly and not args.InputFileList:
        process_images(None, args.det, args.desc, args.kmax, args.Parallel)

    if not args.InputFileList:
        name_to_path_dict, file_list = find_images(image_directories_list, dataset_name_path)
        print('total images: ', len(file_list))

        if args.FindOnly:
            exit(0)
    else:
        with open(args.inputFileList,'r') as fp:
            file_list = fp.readlines()

    process_images(file_list, args.det, args.desc, args.kmax, False)
