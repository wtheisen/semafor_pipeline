import os, faiss, json

import numpy as np

from feature_extractor import feature_detection_and_description
import indexfunctions import sanitize

class Index:
    preproc_str = "OPQ8_32"
    ivf_str = "IVF256"
    pqflat_str = "PQ8"

    ngpu = 1
    nprobe = 24
    tempmem = -1
    ncent = 100

    useNEHST = True
    desc = 'SURF3'
    det = 'SURF3'

    def __init__(self, cache_dir=None):
        self.preproc = None
        self.coarse_quantizer = None
        self.trained_index = None

        self.ID_to_path = {}

        self.cache_dir = cache_dir
        if self.cache_dir:
            self.preproc_file = os.path.join(self.cache_dir, 'preproc.cache')
            self.coarse_quant_file = os.path.join(self.cache_dir, 'coarse_quantizer.cache')
            self.trained_index_file = os.path.join(self.cache_dir, 'trained_index.cache')
            self.ID_to_path_file = os.path.join(self.cache_dir, 'ID_to_path.cache')

        self.img_count = 0

    def extract_features(image_path):
        keyps, feat, det_t, dsc_t = feature_detection_and_description(image_path, 'SURF3', 'SURF3')
        return keyps

    def train_index(self, image_list):
        feature_list = []
        for image_path in image_list:
            feature_list.append = extract_features(image_path)

        preproc = None
        coarse_quantizer = None
        trained_index = None

        if self.preproc_str.startswith('OPQ'):
            fi = preproc_str_local[3:].split('_')
            m = int(fi[0]) #number of subspaces decomposed (subspaces_outputdimension)
            dout = int(fi[1]) if len(fi) == 2 else d #output dimension should be a multiple of the number of subspaces?
            preproc = faiss.OPQMatrix(d, m, dout)
            preproc.train(sanitize(xt_local))

        # Train the coarse quantizer centroids
        if preproc:
            nt = max(10000, 256 * self.ncent)
            d = preproc.d_out
            clus = faiss.Clustering(d, self.ncent)
            clus.verbose = True
            clus.max_points_per_centroid = 10000000

            x = preproc.apply_py(sanitize(feature_list[:nt]))
            vres, vdev = indexfunctions.make_vres_vdev(self.gpu_resources,ngpu=self.ngpu)
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, faiss.IndexFlatL2(d))
            clus.train(x, index)
            centroids = faiss.vector_float_to_array(clus.centroids).reshape(self.ncent, d)

            coarse_quantizer = faiss.IndexFlatL2(preproc.d_out)
            coarse_quantizer.add(centroids)

        # Train the codebooks for the index model
        if preproc and coarse_quantizer:
            d = preproc.d_out
            m = int(self.pqflat_str[2:])

            trained_index = faiss.IndexIVFPQ(coarse_quantizer, d, self.ncent, m, 8)
            trained_index.own_fields = True
            x = preproc.apply_py(sanitize(feature_list[:self.num_training_images]))
            trained_index.train(x)

        self.preproc = preproc
        self.coarse_quantizer = coarse_quantizer
        self.trained_index = trained_index

        if self.self.cache_dir:
            print('[LOG]: Writing trained index pieces...'
            print('\tWriting trained preprocessor...'
            faiss.write_VectorTransform(preproc, self.preproc_file)
            print('\tWriting trained coarse quantizer...'
            np.save(self.coarse_quant_file, coarse_quantizer)
            print('\tWriting trained index...'
            faiss.write_index(trained_index, self.trained_index_file)

    def add_to_index(self, image_list):
        feature_list = []
        for image_path in image_list:
            self.ID_to_path[np.uint64(self.img_count)] = image_path
            feature_list.append = self.preproc.apply_py(extract_features(image_path))
            self.img_count += 1

        self.gpu_index.add_with_ids(feature_list, self.ID_to_path.keys())

        if self.ID_to_path_file:
            with open(self.ID_to_path_file, 'w+') as f:
                f.write(json.dumps(self.ID_to_path))

    def query_index(self, image_list, recall=10):
        if not self.trained_index:
            print('[LOG]: Reading indicies from cached files...')
            self.preproc = faiss.read_index(self.preproc_file)
            self.trained_index = faiss.read_index(self.trained_index_file)

        print('[LOG]: Extracting image features for querying...')
        feature_list = []
        for image_path in image_list:
            feature_list.append = self.preproc.apply_py(extract_features(image_path))


        print('[LOG]: Querying image features...')
        D, I = self.trained_index.search(feature_list, recall)

        ID_distances = []
        for result_distances, result_IDs in zip(D, I):
            ID_distances.append(zip(result_distances, result_IDs))
        return zip(image_list, ID_distances)

    def batch_query_index():
        pass

    def range_query_index():
        pass

    def queries_to_json(raw_queries):
        if not self.ID_to_path:
            with open(self.ID_to_path_file) as f:
                self.ID_to_path = json.load(f)

        pass
