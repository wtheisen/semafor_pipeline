#!/bin/bash

export PYTHONPATH=../notredame/:../featureExtraction/:../indexConstruction/:../provenanceFiltering:../provenanceGraphConstruction:../helperLibraries/:$PYTHONPATH

index_sample=5000
retrieval_recall=1000
motif_clusters=150

datasetName=reddit_clusters

imageRoot1=/afs/crc.nd.edu/user/w/wtheisen/reddit_dataset

featureFolder=/afs/crc.nd.edu/user/w/wtheisen/reddit_semafor_output/features_$datasetName
indexSavePath=/afs/crc.nd.edu/user/w/wtheisen/reddit_semafor_output

#Build list of images
python3 generateServerDictionary.py $imageRoot1 "${indexSavePath}/${datasetName}" &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on generateServerDictionary"
    exit 1
fi

#Feature Extraction
python3 featureExtractionDriver.py --inputFileList "${indexSavePath}/${datasetName}_filelist.txt" --outputdir $featureFolder --PE 6 --det SURF3 --desc SURF3 --kmax 5000 --jobNum 0 --numJobs 1 &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on featureExtractionDriver.py"
    exit 1
fi

#Feature Directory Indexing
python3 generateServerDictionary.py $featureFolder "${indexSavePath}/${datasetName}_features" &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on generateServerDictionary.py round 2"
    exit 1
fi

#Index Training
python3 indexTrainingDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/indextraining_${datasetName}/parameters" &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on indexTrainingDriver.py"
    exit 1
fi

#Index Construction
python3 indexConstructionDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/index_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --GPUCount 1 &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on indexConstructionDriver.py"
    exit 1
fi

python3 generateRandomProbeList.py --ImageList "${indexSavePath}/${datasetName}_filelist.txt" --OutputProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --NumSamples $index_sample &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on generateRandomProbeList.py"
    exit 1
fi

#Provenance Filtering
python3 provenanceFilteringDriver.py --GenericProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --IndexOutputDir "${indexSavePath}/index_${datasetName}/" --ProvenanceOutputFile "${indexSavePath}/results_${datasetName}/results.csv" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --NISTDataset $datasetName --det SURF3 --desc SURF3 --outputdir $indexSavePath --Recall $retrieval_recall &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on provenanceFilteringDriver.py"
    exit 1
fi

#Motif Clustering
python3 ../provenanceFiltering/GraphClustering_new.py --FilteringResultFolder "${indexSavePath}/results_${datasetName}/json" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --ImageRoot $imageRoot1 --ImageFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --OutputFolder "${indexSavePath}/results_${datasetName}/motifClusters" --k $motif_clusters &>> $1
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on GraphClustering_new.py"
    exit 1
fi

#Cluster Visualization
python3 ../helperLibraries/visualizeFilterResultsNew.py --clusters "${indexSavePath}/results_${datasetName}/motifClusters" --folder --datasetName ${datasetName} --filedict "{$indexSavePath}/{$datasetName}_pathmap.json" --outputDir "${indexSavePath}/results_${datasetName}/clustervis" --recall $retrieval_recall
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on visualizeFilterResultsNew.py"
    exit 1
fi
