#!/bin/bash

codeSourcePath=/home/jbrogan4/Documents/Projects/Medifor/GPU_Prov_Filtering_fix/GPU_Prov_Filtering/provenance/tutorial
cd $codeSourcePath

export PYTHONPATH=../notredame/:../featureExtraction/:../indexConstruction/:../provenanceFiltering:../provenanceGraphConstruction:../helperLibraries/:$PYTHONPATH
export PYTHONPATH=/usr/local/lib/python3.7/site-packages/cv2:$PYTHONPATH

index_sample=10000
retrieval_recall=1000
motif_clusters=100

datasetName=4chan_vgg
imageRoot=/media/jbrogan4/bill/Pictures/4chanImages/chanImageScraper/chanImages
# iamgeRoot2=/media/jbrogan4/bill/Pictures/TwitterImages
featureFolder=/media/wtheisen/scratch2/4chan_vgg/features_$datasetName
indexSavePath=/media/wtheisen/scratch2/4chan_vgg


#Build list of images
python3 generateServerDictionary.py $imageRoot "${indexSavePath}/${datasetName}" # this script takes as arguments <path to root of images> <name to save the list of images to>, and outputs 2 files: $datasetName_filelist.txt and $datasetName_pathmap.json

#Feature Extraction
python3 featureExtractionDriver.py --inputFileList "${indexSavePath}/${datasetName}_filelist.txt" --outputdir $featureFolder --PE 5 --det VGG --desc VGG --kmax 5000 --jobNum 0 --numJobs 1

#Feature Directory Indexing
python3 generateServerDictionary.py $featureFolder "${indexSavePath}/${datasetName}_features"

#Index Training
python3 indexTrainingDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/indextraining_${datasetName}/parameters"

#Index Construction
python3 indexConstructionDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/index_${datasetName}" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --GPUCount 1
python3 generateRandomProbeList.py --ImageList "${indexSavePath}/${datasetName}_filelist.txt" --OutputProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --NumSamples $index_sample

#Provenance Filtering
python3 provenanceFilteringDriver.py --GenericProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --IndexOutputDir "${indexSavePath}/index_${datasetName}/" --ProvenanceOutputFile "${indexSavePath}/results_${datasetName}/results.csv" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --NISTDataset $datasetName --Recall $retrieval_recall

#Motif Clustering
python3 ../provenanceFiltering/GraphClustering.py --FilteringResultFolder "${indexSavePath}/results_${datasetName}/json" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --ImageRoot $imageRoot --ImageFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --OutputFolder "${indexSavePath}/results_${datasetName}/motifClusters"

