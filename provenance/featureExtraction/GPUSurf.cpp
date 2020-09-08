void processWithGpu(string objectInputFile, string sceneInputFile, string outputFile, int minHessian = 100)
{
	// Load the image from the disk
	Mat img_object = imread( objectInputFile, IMREAD_GRAYSCALE ); // surf works only with grayscale images
	Mat img_scene = imread( sceneInputFile, IMREAD_GRAYSCALE );
	if( !img_object.data || !img_scene.data ) {
		std::cout<< "Error reading images." << std::endl;
		return;
	}
 
	// Copy the image into GPU memory
	cuda::GpuMat img_object_Gpu( img_object );
	cuda::GpuMat img_scene_Gpu( img_scene );
 
	// Start the timer - the time moving data between GPU and CPU is added
	GpuTimer timer;
	timer.Start();
 
	cuda::GpuMat keypoints_scene_Gpu, keypoints_object_Gpu; // keypoints
	cuda::GpuMat descriptors_scene_Gpu, descriptors_object_Gpu; // descriptors (features)
 
	//-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
	cuda::SURF_CUDA surf( minHessian );
	surf( img_object_Gpu, cuda::GpuMat(), keypoints_object_Gpu, descriptors_object_Gpu );
	surf( img_scene_Gpu, cuda::GpuMat(), keypoints_scene_Gpu, descriptors_scene_Gpu );
	//cout << "FOUND " << keypoints_object_Gpu.cols << " keypoints on object image" << endl;
	//cout << "Found " << keypoints_scene_Gpu.cols << " keypoints on scene image" << endl;
 
	//-- Step 3: Matching descriptor vectors using BruteForceMatcher
	Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
	vector< vector< DMatch> > matches;
	matcher->knnMatch(descriptors_object_Gpu, descriptors_scene_Gpu, matches, 2);
 
	// Downloading results  Gpu -> Cpu
	vector< KeyPoint > keypoints_scene, keypoints_object;
	//vector< float> descriptors_scene, descriptors_object;
	surf.downloadKeypoints(keypoints_scene_Gpu, keypoints_scene);
	surf.downloadKeypoints(keypoints_object_Gpu, keypoints_object);
	//surf.downloadDescriptors(descriptors_scene_Gpu, descriptors_scene);
	//surf.downloadDescriptors(descriptors_object_Gpu, descriptors_object);
 
	timer.Stop();
	printf( "Method processImage() ran in: %f msecs.\n", timer.Elapsed() );
 
	//-- Step 4: Select only goot matches
	//vector<Point2f> obj, scene;
	std::vector< DMatch > good_matches;
	for (int k = 0; k < std::min(keypoints_object.size()-1, matches.size()); k++)
	{
		if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
				((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
		{
			// take the first result only if its distance is smaller than 0.6*second_best_dist
			// that means this descriptor is ignored if the second distance is bigger or of similar
			good_matches.push_back(matches[k][0]);
		}
	}
 
	//-- Step 5: Draw lines between the good matching points
	Mat img_matches;
	drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
			good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
			vector<char>(), DrawMatchesFlags::DEFAULT );
 
	//-- Step 6: Localize the object inside the scene image with a square
	localizeInImage( good_matches, keypoints_object, keypoints_scene, img_object, img_matches );
 
	//-- Step 7: Show/save matches
	//imshow("Good Matches & Object detection", img_matches);
	//waitKey(0);
	imwrite(outputFile, img_matches);
 
	//-- Step 8: Release objects from the GPU memory
	surf.releaseMemory();
	matcher.release();
	img_object_Gpu.release();
	img_scene_Gpu.release();
}
