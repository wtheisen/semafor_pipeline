import sys, cv2, imagehash, time
import numpy as np
import tensorflow as tf

hessianThreshold = 100
nOctaves = 6
nOctaveLayers = 5
extended = False
upright = False
keepTopNCount = 2500
distanceThreshold = 50

def feature_detection(imgpath, img, detetype, kmax=500):
    try:
        if detetype == "SURF3":
            st_t = time.time()
            # detects the SURF keypoints, with very low Hessian threshold
            surfDetectorDescriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=10, nOctaves=nOctaves, nOctaveLayers=nOctaveLayers,
                                                   extended=extended, upright=upright)
            keypoints = surfDetectorDescriptor.detect(img)

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
                        break;

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

    except:
        print("Failure in detecting the keypoints")
        sys.stdout.flush()
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return [], -1

    det_t = ed_t - st_t
    return keypoints, det_t

def feature_description(img, kp, desc_type):
    new_kp = []

    try:
        if desc_type == "SURF3":
            surf = cv2.xfeatures2d.SURF_create()
            st_t = time.time()
            __, features = surf.compute(img, kp)
            ed_t = time.time()

        elif desctype == "PHASH":
            st_t = time.time()
            features = imagehash.phash(img)
            hFeatures = [float(ord(c) * 0.001) for c in list(str(features))]
            features = np.array([hFeatures])
            print(features.shape)
            ed_t = time.time()

        elif desctype == "VGG":
            # print('DOIN DAT VGG BB')
            import tensorflow as tf
            from keras import backend as kBackend
            from keras.preprocessing import image
            from keras.applications.vgg19 import VGG19
            from keras.applications.vgg19 import preprocess_input

            core_config = tf.ConfigProto()
            core_config.gpu_options.allow_growth = True
            session = tf.Session(config=core_config)
            kBackend.set_session(session)

            m = VGG19(weights='imagenet', include_top=False, pooling='avg')
            # m = VGG19(weights='imagenet', include_top=False)

            st_t = time.time()

            i = image.load_img(imgpath, target_size=(244, 244))
            i_data = image.img_to_array(i)
            i_data = np.expand_dims(i_data, axis=0)
            i_data = preprocess_input(i_data)

            features = m.predict(i_data)

            ed_t = time.time()

            session.close()
            kBackend.clear_session()
    except:
        print("Failure in describing the keypoints")
        sys.stdout.flush()
        e_type, e_val, e_tb = sys.exc_info()
        traceback.print_exception(e_type, e_val, e_tb)
        return [], -1, 0,[]

    dsc_t = ed_t - st_t
    return features, dsc_t, 1, new_kp

def feature_detection_and_description(imgpath, detetype, desctype, kmax=500, img=[]):
    keyps = None
    det_t=0

    keyps, det_t = feature_detection(imgpath, img, detetype, kmax,)

    if not keyps:
        print('[ERROR]: Failed to detect keypoints for {img_path}')
        return None, None, None, None

    feat, dsc_t, success, keyps2 = feature_description(img, keyps, desctype)

    if keyps is None or len(keyps) == 0:
        keyps= keyps2

    if keyps == []:
        keyps = keyps2
    if feat is []:
        return keyps, [], None, None

    return keyps, feat, det_t, dsc_t
