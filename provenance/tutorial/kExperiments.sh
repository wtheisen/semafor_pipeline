codeSourcePath=/home/jbrogan4/Documents/Projects/Medifor/GPU_Prov_Filtering_fix/GPU_Prov_Filtering/provenance/tutorial
cd $codeSourcePath

export PYTHONPATH=../notredame/:../featureExtraction/:../indexConstruction/:../provenanceFiltering:../provenanceGraphConstruction:../helperLibraries/:$PYTHONPATH
export PYTHONPATH=/usr/local/lib/python3.7/site-packages/cv2:$PYTHONPATH

datasetName=phash_clusters
imageRoot1=/media/wtheisen/scratch2/chosenClusterImages/
indexSavePath=/media/wtheisen/scratch2/indo_phash

rm /media/wtheisen/scratch2/indo_phash/results_phash_clusters/motifClusters/*.json

for i in {10..400..65}; do
    python3 ../provenanceFiltering/GraphClustering_new.py \
        --FilteringResultFolder "${indexSavePath}/results_${datasetName}/json" \
        --CacheFolder "${indexSavePath}/cache_${datasetName}/" \
        --ImageRoot $imageRoot1 \
        --ImageFileList "${indexSavePath}/${datasetName}_features_filelist.txt" \
        --OutputFolder "${indexSavePath}/results_${datasetName}/motifClusters" \
        --k $i

    RETVAL=$?
    if [ $RETVAL -ne 0 ]; then
        echo "Failure on GraphClustering_new.py"
        exit 1
    fi

    ls /media/wtheisen/scratch2/indo_phash/results_phash_clusters/motifClusters/ \
        | cut -d '_' -f 3 \
        | sed 's/size//' \
        | sed 's/.json//' > indo_phash_${i}_sizes.txt

    rm /media/wtheisen/scratch2/indo_phash/results_phash_clusters/motifClusters/*.json
done
