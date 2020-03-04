#!bash

function check_appid_valid() {
    appid="$1"
    num=`echo "${appid}" |awk -F '-' '{print NF}'`
    if [ $num -ne 4 ];then
        return 1
    fi
    return 0
}

function appid_running_num() {
    appid="$1"
    proc_num=`ps -ef |grep "${appid}"|grep -v grep|wc -l`
    if [ $? -ne 0 ];then 
        #if failed, return 1, avoid
        return 1
    fi
    return ${proc_num}
}

work_dir="$1"
base_dir=`echo "${work_dir}" |awk -F 'app-user-' '{print $1}'`
database_list=`find ${base_dir} -type d -name 'database'`
for element in ${database_list[@]}
do
    app_id=`echo "$element"|awk -F 'app-user-' '{print $2}' |awk -F '/' '{print "app-user-"$1}'`
    check_appid_valid "${app_id}"
    if [ $? -ne 0 ];then
        continue
    fi
    appid_running_num "${app_id}"
    if [ $? -eq 0 ];then
        echo "remove ${element}"
        rm -rf ${element}
    fi
done

