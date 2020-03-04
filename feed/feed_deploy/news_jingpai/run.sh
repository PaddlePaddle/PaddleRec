#!/bin/sh
source ~/.bashrc

# Author: Wu.ZG
# Created Time : 2017-08-14 21:31:56
# File Name: guard.sh
# Version:
# Description:
# Last modified: 2018-01-29 11:00:42

set -x

SLEEP=10
HOST=`hostname`
WORKROOT=${PWD}
RUN_SCRIPT="${WORKROOT}/submit.sh"
#ALARM="./alarm.sh"
on_duty=(
    # RD
    # OP
    # QA
    15101120768
)

function alarm() {
    content=$1
    for phone_num in ${on_duty[@]};do
        echo ${phone_num} ${content}
        gsmsend -s emp01.baidu.com:15001 "${phone_num}"@"$1"
    done
    echo "$1" | mail -s "$1" $email
}

pid=$$
echo ${pid} > pid

if [ ! -d "./log" ];then
    mkdir log
fi

while [ 1 ]
do
    sh ${RUN_SCRIPT} > log/"`date +"%Y%m%d_%H%M%S"`".log
    RET=$?

    #source ${ALARM}
    if [ ${RET} -ne 0 ];then
       content="`date +"%Y%m%d %H:%M:%S "` Job fail. Exit ${RET}. From ${HOST}:${WORKROOT}. Pid=${pid}"
       echo "${content}"
       alarm "${content}"
    else
       content="`date +"%Y%m%d %H:%M:%S "` Job finish. From ${HOST}:${WORKROOT}. Pid=${pid}"
       echo "${content}"
       alarm "${content}"
       break
    fi

    sleep ${SLEEP}

done

echo "`date +"%Y%m%d %H:%M:%S "` guard exit."
