import os, faiss, json

import numpy as np

from tqdm import tqdm
from inspect import currentframe, getframeinfo
from debug import d_print

from read_fvecs import load_sift_features
from feature_extractor import parallel_feature_extraction
# from utils import nhs_filter, queries_to_json, write_query_results

class Index:
    preproc_str = "OPQ16_64"
    ivf_str = "IVF256"
    pqflat_str = "PQ16"

    ngpu = 1
    nprobe = 24
    tempmem = -1
    ncent = 256

    useNEHST = True
    desc = 'SURF3'
    det = 'SURF3'

    def __init__(self, gpu=True, cache_dir=None):
        self.preproc = None
        self.coarse_quantizer = None
        self.trained_index = None

        self.query_img_data = None

        self.ID_counter = 0
        self.ID_list = []
        self.feature_to_ID = {}
        self.ID_to_path = {}

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.preproc_file = os.path.join(self.cache_dir, 'preproc.cache')
            self.coarse_quant_file = os.path.join(self.cache_dir, 'coarse_quantizer.cache')
            self.trained_index_file = os.path.join(self.cache_dir, 'trained_index.cache')
            self.ID_to_path_file = os.path.join(self.cache_dir, 'ID_to_path.cache')

        self.gpu = gpu
        if self.gpu:
            self.res = faiss.StandardGpuResources()

    def ID_features(self, img_path, img_data_dict):
        for dummy_id, feature in img_data_dict['feature_dict'].items():
            id_labelled_feature_dict = {}

            self.ID_list.append(self.ID_counter)
            self.ID_to_path[self.ID_counter] = img_path
            id_labelled_feature_dict[self.ID_counter] = feature
            img_data_dict['feature_dict'] = id_labelled_feature_dict.copy()

            self.ID_counter += 1

    def features_from_path_list(self, path_list, ID=False):
        d_print('LOG', getframeinfo(currentframe()), 'Extracting image features from a list')

        feature_list = np.array([], dtype=np.float32).reshape((0, 64))

        all_img_data_dict, raw_feature_list = parallel_feature_extraction(path_list)
        d_print('LOG', getframeinfo(currentframe()), 'Feature extraction complete')

        if ID:
            d_print('LOG', getframeinfo(currentframe()), 'IDing images')
            for img_path, img_data_dict in tqdm(all_img_data_dict.items()):
                self.ID_features(img_path, img_data_dict)

            self.query_img_data = all_img_data_dict

        d_print('LOG', getframeinfo(currentframe()), 'Concatenating features for training')
        feature_list = np.concatenate(raw_feature_list)

        return feature_list

    def train_index(self, image_list):
        # training_features = self.features_from_path_list(image_list)
        training_features = load_sift_features('/media/wtheisen/scratch2/siftsmall/siftsmall_learn.fvecs')
        # print(training_features)

        preproc = None
        coarse_quantizer = None
        trained_index = None

        d_print('LOG', getframeinfo(currentframe()), 'Training preprocessor')
        if self.preproc_str.startswith('OPQ'):
            d = 128
            fi = self.preproc_str[3:].split('_')
            m = int(fi[0]) #number of subspaces decomposed (subspaces_outputdimension)
            dout = int(fi[1]) if len(fi) == 2 else d #output dimension should be a multiple of the number of subspaces?
            preproc = faiss.OPQMatrix(d, m, dout)

            if self.gpu:
                preproc = faiss.index_cpu_to_gpu(self.res, 0, preproc)

            preproc.train(training_features)

        d_print('LOG', getframeinfo(currentframe()), 'Training coarse quantizer')
        # Train the coarse quantizer centroids
        if preproc:
            nt = max(10000, 256 * self.ncent)
            d = preproc.d_out
            clus = faiss.Clustering(d, self.ncent)
            clus.verbose = True
            clus.max_points_per_centroid = 10000000

            x = preproc.apply_py(training_features)
            index = faiss.IndexFlatL2(d)
            clus.train(x, index)
            centroids = faiss.vector_float_to_array(clus.centroids).reshape(self.ncent, d)

            coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
            coarse_quantizer.add(centroids)

        d_print('LOG', getframeinfo(currentframe()), 'Training index')
        # Train the codebooks for the index model
        if preproc and coarse_quantizer:
            d = preproc.d_out
            m = int(self.pqflat_str[2:])

            trained_index = faiss.IndexIVFPQ(coarse_quantizer, d, self.ncent, m, 8)

            if self.gpu:
                trained_index = faiss.index_cpu_to_gpu(self.res, 0, trained_index)

            trained_index.own_fields = True
            x = preproc.apply_py(training_features)
            trained_index.train(x)

        self.preproc = preproc
        self.coarse_quantizer = coarse_quantizer
        self.trained_index = trained_index

        if self.cache_dir:
            d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing trained preprocessor')
            faiss.write_VectorTransform(preproc, self.preproc_file)
            d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing coarse quantizer')
            np.save(self.coarse_quant_file, coarse_quantizer)
            d_print('LOG', getframeinfo(currentframe()), 'Writing index cache files', l2='Writing trained index')
            faiss.write_index(trained_index, self.trained_index_file)

    def add_to_index(self, image_list):
        # feature_list = self.features_from_path_list(image_list, ID=True)
        feature_list = load_sift_features('/media/wtheisen/scratch2/siftsmall/siftsmall_base.fvecs')

        count = 0
        self.id_labelled_feature_dict = {}
        for i in feature_list:
            self.ID_list.append(count)
            self.id_labelled_feature_dict[count] = i
            count += 1

        d_print('LOG', getframeinfo(currentframe()), 'Adding images to index')
        self.trained_index.add_with_ids(self.preproc.apply_py(feature_list), np.array(self.ID_list))
        # self.trained_index.add(self.preproc.apply_py(feature_list))

        if self.cache_dir:
            with open(self.ID_to_path_file, 'w+') as f:
                f.write(json.dumps(self.ID_to_path))

    def query_index(self, image_list, recall=100):
        if not self.trained_index:
            d_print('LOG', getframeinfo(currentframe()), 'Reading index from cache')
            self.preproc = faiss.read_index(self.preproc_file)
            self.trained_index = faiss.read_index(self.trained_index_file)

        self.recall = recall

        d_print('LOG', getframeinfo(currentframe()), 'Extracting image features for querying')
        # query_features = self.features_from_path_list(image_list[:5])
        query_features = load_sift_features('/media/wtheisen/scratch2/siftsmall/siftsmall_query.fvecs')

        d_print('LOG', getframeinfo(currentframe()), 'Querying image features')
        D, I = self.trained_index.search(self.preproc.apply_py(query_features), self.recall)

        d_print('LOG', getframeinfo(currentframe()), 'Mapping query result IDs to image paths')
        ID_distances = []
        for result_distances, result_IDs in tqdm(zip(D, I)):
            ID_distances.append(list(zip(result_distances, result_IDs)))

        for q_r in ID_distances:
            fails = 0
            for r in q_r:
                if r[1] == -1:
                    fails += 1
            print('Fails:', fails)

        # return ID_distances
        return []

        # return self.queries_to_json(list(zip(image_list, ID_distances)))

    def queries_to_json(self, raw_queries):
        print(raw_queries)
        return None

        query_dict = {}
        for query_image_path, results in raw_queries.items():
            results_dict = {}
            for distance, ID in results.items():
                results_dict[distance] = self.ID_to_path[ID]

            query_dict[query_image_path] = results_dict

        return query_dict
