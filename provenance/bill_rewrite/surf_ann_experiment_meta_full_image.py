import json, collections, os
import read_fvecs, find_ground_truth

import numpy as np

from tqdm import tqdm
from itertools import chain

from index_class import Index
from feature_extractor import feature_detection_and_description

def prepare_json():
    results_json = {}
    results_json['trained'] = {}
    results_json['dilued'] = {}

    for i in range(0, 97):
        results_json['trained'][str(i)] = []
        results_json['dilued'][str(i)] = []

    return results_json

def feature_to_image_voting(id_to_path, returned_feature_ids):
    voted_images = collections.Counter()

    for result_id in returned_feature_ids[0]:
        voted_images[id_to_path[result_id]] += 1

    return voted_images

def feature_extraction(img_path):
    keyps, feats, det_t, dsc_t = feature_detection_and_description(img_path)
    return feats

def dilute(core, destractor, dilu_amount):
    return np.concatenate([core, destractor[:dilu_amount]])

def compute_diluted_gt(feature_vectors_in_index, ids, query_feature_vectors):
    i = find_ground_truth.build_flat_index(feature_vectors_in_index, ids, dim=64)
    dists, ids = find_ground_truth.find_ground_truth(query_feature_vectors, i, recall=recall)
    return ids

def walk_classes_dirs(root_path):
    class_dir_paths = []
    for i in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, i)):
            class_dir_paths.append(os.path.join(root_path, i))

    return class_dir_paths

# with open('./surf_results_template.json') as f:
#     results_json = json.load(f)
results_json = prepare_json()

recall = 10
n_queries = 100
dataset = 'reddit_photoshop_battles'

load_from_images = True
images_root_path = '/media/wtheisen/scratch2/Reddit_Prov_Dataset_v6/Data/'

id_to_path = {}
feature_to_id = []

training_features = []
feature_list = []
query_features = []

print('Loading dataset', dataset)

img_class_features_dict = {}

if load_from_images:
    img_count = 0
    # with open(image_list_path) as f:
    #     image_path_list = f.readlines().rstrip().lstrip()

    class_dir_paths = walk_classes_dirs(images_root_path)

    for image_class in tqdm(class_dir_paths, desc='Classes'):
        img_class_features_dict[image_class] = {}
        for image in tqdm(os.listdir(image_class), desc='Images'):
            if not image.endswith('.png') and not image.endswith('.jpg'):
                continue

            feats = feature_extraction(os.path.join(image_class, image))

            try:
                if not feats.any():
                    print('Failed to extract features, skipping...')
                    continue
            except:
                print('Failed to extract features, skipping...')
                continue

            ided_feats = []
            for f in feats:
                ided_feats.append((f, img_count))
                feature_to_id.append((f, img_count))

            img_class_features_dict[image_class][image] = ided_feats

            id_to_path[img_count] = image
            img_count += 1

for i in tqdm(range(0, 10), desc='Trials'):
    #select core set from the full feature list
    np.random.shuffle(class_dir_paths)
    rando_classes = np.array_split(class_dir_paths, 2)
    rando_core_classes = rando_classes[0]
    rando_dilu_classes = rando_classes[1]

    rando_core_images = []
    for c in rando_core_classes:
        for i in img_class_features_dict[c]:
            rando_core_images.append(i)
    rando_query_images = np.random.choice(rando_core_images, n_queries)

    rando_core_tuples = []
    for c in rando_dilu_classes:
        for i in img_class_features_dict[c]:
            rando_core_tuples += img_class_features_dict[c][i]

    compounding_dilu_tuples = []

    #now for each distractor class Sigma-add it to the core set
    dilu_level = 1
    for dilu_class in rando_dilu_classes:
        for i in img_class_features_dict[dilu_class]:
            compounding_dilu_tuples += img_class_features_dict[dilu_class][i]

        dilued_tuples = rando_core_tuples + compounding_dilu_tuples

        index_retrain = Index()
        index_dilu = Index()

        index_retrain.train_index(None, training_features=np.asarray([i[0] for i in dilued_tuples]))
        index_dilu.train_index(None, training_features=np.asarray([i[0] for i in rando_core_tuples]))

        #need to get ids
        index_retrain.add_to_index(None,
                           feature_list=np.asarray([i[0] for i in dilued_tuples]),
                           ids=np.asarray([i[1] for i in dilued_tuples]))
        index_dilu.add_to_index(None,
                           feature_list=np.asarray([i[0] for i in dilued_tuples]),
                           ids=np.asarray([i[1] for i in dilued_tuples]))

        #need to get query features, need to query IMAGES not features, for loop?
        query_precision_list_retrain = []
        query_precision_list_dilu = []
        for query_image in tqdm(rando_query_images, desc='Queries'):
            query_tuples = [img_class_features_dict[c][query_image] for c in img_class_features_dict.keys() if query_image in img_class_features_dict[c]]
            query_feature_list = np.asarray([i[0] for i in query_tuples[0]])
            query_id_list = np.asarray([i[1] for i in query_tuples[0]])
            # query_feature_list = np.asarray(list(chain(*query_tuples)))

            dists, ids_core = index_retrain.query_index(None, query_feature_list=query_feature_list, recall=recall)
            voted_images_retrain = feature_to_image_voting(id_to_path, ids_core)

            dists, ids_dilu = index_dilu.query_index(None, query_feature_list=query_feature_list, recall=recall)
            voted_images_dilu = feature_to_image_voting(id_to_path, ids_dilu)

            #compute g_t
            ids_g_t = compute_diluted_gt(np.asarray([i[0] for i in dilued_tuples]),
                                       np.asarray([i[1] for i in dilued_tuples]),
                                       np.asarray([i[0] for i in query_tuples[0]]))
            voted_images_g_t = feature_to_image_voting(id_to_path, ids_g_t)

            #get single precision score for retrained
            query_precision_list_retrain.append(len(list(set(voted_images_retrain).intersection(voted_images_g_t))) / 100)
            #get single precision score for dilued
            query_precision_list_dilu.append(len(list(set(voted_images_dilu).intersection(voted_images_g_t))) / 100)

        results_json['trained'][str(dilu_level)] += query_precision_list_retrain
        results_json['dilued'][str(dilu_level)] += query_precision_list_dilu
        dilu_level += 1

print('Finished')
with open('./surf_class_simu_50d_t100_results', 'w+') as f:
    json.dump(results_json, f)
