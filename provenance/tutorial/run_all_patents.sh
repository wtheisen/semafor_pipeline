#!/bin/bash

codeSourcePath=/afs/crc.nd.edu/user/w/theisen/GPU_Prov_Filtering_fix/GPU_Prov_Filtering/provenance/tutorial
cd $codeSourcePath

export PYTHONPATH=../notredame/:../featureExtraction/:../indexConstruction/:../provenanceFiltering:../provenanceGraphConstruction:../helperLibraries/:$PYTHONPATH
#export PYTHONPATH=/usr/local/lib/python3.7/site-packages/cv2:$PYTHONPATH

index_sample=10000
#index_sample=5000
retrieval_recall=1000
# motif_clusters = 500

datasetName=patent_clusters
imageRoot1=~/patents_two/
# imageRoot1=/media/jbrogan4/bill/Pictures/InstagramImages
# imageRoot2=/media/jbrogan4/bill/Pictures/TwitterImages
# featureFolder=/home/wtheisen/indoClusteringResults/features_$datasetName
# indexSavePath=/home/wtheisen/indoClusteringResults

# featureFolder=/media/wtheisen/scratch2/indo_vgg/features_$datasetName
# indexSavePath=/media/wtheisen/scratch2/indo_vgg
featureFolder=/afs/crc.nd.edu/user/w/wtheisen/patent_cluster_data/features_$datasetName
indexSavePath=/afs/crc.nd.edu/user/w/wtheisen/patent_cluster_data

# featureFolder=/media/wtheisen/scratch2/Indonesia_Retry/features_$datasetName
# indexSavePath=/media/wtheisen/scratch2/Indonesia_Retry


#Build list of images
#python3 generateServerDictionary.py $imageRoot1,$imageRoot2 "${indexSavePath}/${datasetName}" # this script takes as arguments <path to root of images> <name to save the list of images to>, and outputs 2 files: $datasetName_filelist.txt and $datasetName_pathmap.json

python3 generateServerDictionary.py $imageRoot1 "${indexSavePath}/${datasetName}" # this script takes as arguments <path to root of images> <name to save the list of images to>, and outputs 2 files: $datasetName_filelist.txt and $datasetName_pathmap.json
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on generateServerDictionary"
    exit 1
fi

exit

#feature Extraction
python3 featureExtractionDriver.py --inputFileList "${indexSavePath}/${datasetName}_filelist.txt" --outputdir $featureFolder --PE 12 --det SURF3 --desc SURF3 --kmax 5000 --jobNum 0 --numJobs 1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on featureExtractionDriver.py"
    exit 1
fi

#feature Directory Indexing
python3 generateServerDictionary.py $featureFolder "${indexSavePath}/${datasetName}_features"
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on generateServerDictionary.py"
    exit 1
fi

#index Training
python3 indexTrainingDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/indextraining_${datasetName}/parameters"
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on indexTrainingDriver.py"
    exit 1
fi

#index Construction
python3 indexConstructionDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/index_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --GPUCount 1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on indexConstructionDriver.py"
    exit 1
fi

# python3 generateRandomProbeList.py --ImageList "${indexSavePath}/${datasetName}_filelist.txt" --OutputProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --NumSamples $index_sample
# RETVAL=$?
# if [ $RETVAL -ne 0 ]; then
#     echo "Failure on generateRandomProbeList.py"
#     exit 1
# fi

#Provenance Filtering
# python3 provenanceFilteringDriver.py --GenericProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --IndexOutputDir "${indexSavePath}/index_${datasetName}/" --ProvenanceOutputFile "${indexSavePath}/results_${datasetName}/results.csv" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --NISTDataset $datasetName --det PHASH --desc PHASH --outputdir $indexSavePath --Recall $retrieval_recall
# RETVAL=$?
# if [ $RETVAL -ne 0 ]; then
#     echo "Failure on provenanceFilteringDriver.py"
#     exit 1
# fi

#Motif Clustering
#python3 ../provenanceFiltering/GraphClustering_new.py --FilteringResultFolder "${indexSavePath}/results_${datasetName}/json" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --ImageRoot $imageRoot1 --ImageFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --OutputFolder "${indexSavePath}/results_${datasetName}/motifClusters" --k 150
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on GraphClustering_new.py"
#    exit 1
#fi
