#!/bin/bash

feature_type="SURF3"
index_sample=5000
retrieval_recall=1000
motif_clusters=150

dataset_name=pro_publica_b
imageRoot1=/nfs/mews_extract/pro_publica
output_dir=/home/pthomas4/semafor/semafor_output

#Build list of images
python3 find_extract_images.py --ImageDirectoryList $imageRoot1 \
                                --OutputDir "${output_dir}" \
                                --DatasetName "${dataset_name}" \
                                --PE 6 --TFCores 6 \
                                --det "${feature_type}" --desc "${feature_type}"
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    echo "Failure on finding images and extracting features"
    exit 1
fi

#Index Training
#python3 indexTrainingDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/indextraining_${datasetName}/parameters" &>> $1
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on indexTrainingDriver.py"
#    exit 1
#fi

#Index Construction
#python3 indexConstructionDriver.py --FeatureFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --IndexOutputFile "${indexSavePath}/index_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --GPUCount 1 &>> $1
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on indexConstructionDriver.py"
#    exit 1
#fi

#python3 generateRandomProbeList.py --ImageList "${indexSavePath}/${datasetName}_filelist.txt" --OutputProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --NumSamples $index_sample &>> $1
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on generateRandomProbeList.py"
#    exit 1
#fi

#Provenance Filtering
#python3 provenanceFilteringDriver.py --GenericProbeList "${indexSavePath}/${datasetName}_randomProbes.txt" --IndexOutputDir "${indexSavePath}/index_${datasetName}/" --ProvenanceOutputFile "${indexSavePath}/results_${datasetName}/results.csv" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --TrainedIndexParams "${indexSavePath}/indextraining_${datasetName}/parameters" --NISTDataset $imageRoot1 --det SURF3 --desc SURF3 --outputdir $indexSavePath --Recall $retrieval_recall
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on provenanceFilteringDriver.py"
#    exit 1
#fi

#Motif Clustering
python3 ../provenanceFiltering/GraphClustering_new.py --FilteringResultFolder "${indexSavePath}/results_${datasetName}/json" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --ImageRoot $imageRoot1 --ImageFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --OutputFolder "${indexSavePath}/results_${datasetName}/motifClusters" --k $motif_clusters --Algorithm Markov
#python3 ../provenanceFiltering/GraphClustering_new.py --FilteringResultFolder "${indexSavePath}/results_${datasetName}/json" --CacheFolder "${indexSavePath}/cache_${datasetName}/" --ImageRoot $imageRoot1 --ImageFileList "${indexSavePath}/${datasetName}_features_filelist.txt" --OutputFolder "${indexSavePath}/results_${datasetName}/motifClusters" --k $motif_clusters --Algorithm Markov &>> $1
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on GraphClustering_new.py"
#    exit 1
#fi

#Cluster Visualization
python3 ../helperLibraries/visualizeFilterResultsNew.py --clusters "${indexSavePath}/results_${datasetName}/motifClusters" --folder ${imagelink_folder} --datasetName ${datasetName} --filedict "${indexSavePath}/${datasetName}_pathmap.json" --outputDir "${indexSavePath}/results_${datasetName}/clustervis" --recall $retrieval_recall
#RETVAL=$?
#if [ $RETVAL -ne 0 ]; then
#    echo "Failure on visualizeFilterResultsNew.py"
#    exit 1
#fi
