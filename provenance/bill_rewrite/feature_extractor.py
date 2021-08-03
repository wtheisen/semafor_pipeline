import sys, cv2, imagehash, time, os
import concurrent.futures
import copyreg
import numpy as np

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="2"

hessianThreshold = 100
nOctaves = 6
nOctaveLayers = 5
extended = False
upright = False
keepTopNCount = 2500
distanceThreshold = 50

#python magic to write custom pickle method for cv2.KeyPoints:
#    https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror/48832618
def _pickle_keypoints(point):
    return cv2.KeyPoint, (*point.pt, point.size, point.angle,
                            point.response, point.octave, point.class_id)
copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoints)

def draw_keypoint_matches(img_1, img_2):
    kp_1, desc_1, _, _ = feature_detection_and_description(img_1, gpu=False)
    kp_2, desc_2, _, _ = feature_detection_and_description(img_2, gpu=False)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(desc_1, desc_2)
    matches = sorted(matches, key=lambda x:x.distance)

    img_1 = cv2.imread(img_1)
    img_2 = cv2.imread(img_2)
    match_img = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches[:50], img_2, flags=2)

    return match_img

def extract_image_features(image_path, det='SURF3', desc='SURF3'):
    keypoints, features, w, h = feature_detection_and_description(image_path, det, desc)

    # feature_dict = {}
    # dummy_id = 0

    # for feature in features:
    #     feature_dict[str(dummy_id)] = feature
    #     dummy_id += 1

    # return {'keypoints': keypoints, 'feature_dict': feature_dict, 'img_size': (w, h)}
    return {'keypoints': keypoints, 'feature_dict': dict(zip(np.arange(0, 501, 1), features)), 'img_size': (w, h)}

def parallel_feature_extraction(image_path_list, PE=6, det='SURF3', desc='SURF3'):
    feature_dict = {}
    raw_feature_list = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=PE) as executor:
        for image_path, img_data_dict in tqdm(zip(image_path_list, executor.map(extract_image_features, image_path_list)), total=len(image_path_list)):
            feature_dict[image_path] = img_data_dict
            raw_feature_list.append(list(img_data_dict['feature_dict'].values()))

    return feature_dict, raw_feature_list

def feature_detection(imgpath, img, detetype, gpu=True, kmax=500):
    try:
        if detetype == "SURF3":
            st_t = time.time()
            # detects the SURF keypoints, with very low Hessian threshold

            if not gpu:
                surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                       extended=extended, upright=upright)
                keypoints = surf.detect(img)
            else:
                surf_gpu = cv2.cuda.SURF_CUDA_create(_hessianThreshold=10, _nOctaves=nOctaves, _nOctaveLayers=nOctaveLayers,
                                                       _extended=extended, _upright=upright)

                def upscale_image(img):
                    return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_AREA)

                detect_worked = False
                while not detect_worked:
                    try:
                        gpu_img = cv2.cuda_GpuMat()
                        gpu_img.upload(img)
                        keypoints_gpu = surf_gpu.detect(gpu_img, None)
                    except:
                        img = upscale_image(img)
                    else:
                        detect_worked = True

                keypoints = cv2.cuda_SURF_CUDA.downloadKeypoints(surf_gpu, keypoints_gpu)

            # sorts the keypoints according to their Hessian value
            keypoints = sorted(keypoints, key=lambda match: match.response, reverse=True)

            # obtains the positions of the keypoints within the described image
            positions = []
            for kp in keypoints:
                positions.append((kp.pt[0], kp.pt[1]))
            positions = np.array(positions).astype(np.float32)

            # selects the keypoints based on their positions and distances
            selectedKeypoints = []
            selectedPositions = []

            if len(keypoints) > 0:
                # keeps the top-n strongest keypoints
                for i in range(min(keepTopNCount,len(keypoints))):
                    selectedKeypoints.append(keypoints[i])
                    selectedPositions.append(positions[i])

                    # if the amount of wanted keypoints was reached, quits the loop
                    if len(selectedKeypoints) >= kmax:
                        break

                selectedPositions = np.array(selectedPositions)

                # adds the remaining keypoints according to the distance threshold,
                # if the amount of wanted keypoints was not reached yet
                # print('selected keypoints size: ', len(selectedKeypoints), ' kmax: ',kmax)
                if len(selectedKeypoints) < kmax:
                    matcher = cv2.BFMatcher()
                    for i in range(keepTopNCount, positions.shape[0]):
                        currentPosition = [positions[i]]
                        currentPosition = np.array(currentPosition)

                        match = matcher.match(currentPosition, selectedPositions)[0]
                        if match.distance > distanceThreshold:
                            selectedKeypoints.append(keypoints[i])
                            selectedPositions = np.vstack((selectedPositions, currentPosition))

                        # if the amount of wanted keypoints was reached, quits the loop
                        if len(selectedKeypoints) >= kmax:
                            break;
                keypoints = selectedKeypoints
            ed_t = time.time()

        elif detetype == "PHASH":
            st_t = time.time()
            h, w, d = img.shape
            x_center, y_center = int(w / 2), int(h / 2)
            keypoints = [cv2.KeyPoint(x=x_center, y=y_center, _size=1, _angle=0)]
            ed_t = time.time()

        elif detetype == "VGG":
            st_t = time.time()
            h, w, d = img.shape
            x_center, y_center = int(w / 2), int(h / 2)
            keypoints = [cv2.KeyPoint(x=x_center, y=y_center, _size=1, _angle=0)]
            ed_t = time.time()

    except Exception as e:
        print(e, img.shape)
        # print("[ERROR]: Failure in detecting keypoints...")
        # print(f"\t{e}")
        return [], -1

    det_t = ed_t - st_t
    # print('succccc')
    return keypoints, img

def feature_description(img, kp, desc_type, gpu=True):
    new_kp = []

    # try:
    if desc_type == "SURF3":
        st_t = time.time()
        if not gpu:
            surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                       extended=extended, upright=upright)
            __, features = surf.compute(img, kp)
        else:
            surf_gpu = cv2.cuda.SURF_CUDA_create(_hessianThreshold=10, _nOctaves=nOctaves, _nOctaveLayers=nOctaveLayers,
                                                   _extended=extended, _upright=upright)
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            try:
                features_gpu = surf_gpu.usefulDetectDescribe(gpu_img, mask=None, keypoints=kp, useProvidedKeypoints=True)
                # gpu_img.free()
                features = np.reshape(features_gpu, (-1, 64))
            except:
                pass

        ed_t = time.time()

    elif desc_type == "PHASH":
        st_t = time.time()
        features = imagehash.phash(img)
        hFeatures = [float(ord(c) * 0.001) for c in list(str(features))]
        features = np.array([hFeatures])
        ed_t = time.time()

    elif desc_type == "VGG":
        import tensorflow as tf
        from keras import backend as kBackend
        from keras.preprocessing import image
        from keras.applications.vgg19 import VGG19
        keypoints = surfDetectorDescriptor.detect(img)
        core_config = tf.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.Session(config=core_config)
        kBackend.set_session(session)

        m = VGG19(weights='imagenet', include_top=False, pooling='avg')
        # m = VGG19(weights='imagenet', include_top=False)

        st_t = time.time()

        i = image.load_img(img, target_size=(244, 244))
        i_data = image.img_to_array(i)
        i_data = np.expand_dims(i_data, axis=0)
        i_data = preprocess_input(i_data)

        features = m.predict(i_data)

        ed_t = time.time()

        session.close()
        kBackend.clear_session()
    # except:
    #     print("[ERROR]: Failure in describing keypoints...")
    #     return [], -1, 0,[]

    dsc_t = ed_t - st_t
    return features, dsc_t, 1, new_kp

def feature_detection_and_description(img_path, detetype='SURF3', desctype='SURF3', kmax=500, img=None, gpu=True):
    keyps = None
    det_t=0

    if not img:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # print(img_path)
        # print(img)

    keyps, img = feature_detection(img_path, img, detetype, gpu=gpu, kmax=kmax)

    if not keyps:
        print(f'[ERROR]: Failed to detect keypoints for {img_path}')
        return None, None, None, None

    feat, dsc_t, success, keyps2 = feature_description(img, keyps, desctype, gpu=gpu)

    if keyps is None or len(keyps) == 0:
        keyps= keyps2

    if keyps == []:
        keyps = keyps2
    if feat == []:
        return keyps, [], None, None

    return keyps, feat, det_t, dsc_t
