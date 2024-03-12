#!/bin/bash
base_dir=$1

MAX_WORKERS=100
for file_or_path in ${base_dir}/*; do
    if [ -d ${file_or_path} ]; then
        count=0
        for file in ${file_or_path}/*; do
            filepath=`dirname ${file}`
            filename=${file##*/} 
            prefix=`echo $filename | cut -d '.' -f 1 `
            suffix=${filename##*.}
            if [ ${suffix} = gz ]; then
                echo "[UNZIP] unzip ${file} --> ${filepath}/${prefix}"
                zcat ${file} > ${filepath}/${prefix} &
            fi
            count=`echo $count + 1 | bc`
            if [ $count -eq ${MAX_WORKERS} ]; then
                wait
                count=0
            fi

        done
        wait
        rm -rf ${file_or_path}/*.gz
    else
        filepath=`dirname ${file_or_path}`
        filename=${file_or_path##*/} 
        prefix=`echo $filename | cut -d '.' -f 1 `
        suffix=${filename##*.}
        if [ ${suffix} = gz ]; then
            echo "[UNZIP] unzip ${file_or_path} --> ${filepath}/${prefix}"
            zcat ${file_or_path} > ${filepath}/${prefix}
            rm -rf ${file_or_path}
        fi
    fi
done
