# 解释yaml配置选项并转化为环境变量
# eval $(parse_yaml "$(dirname "${BASH_SOURCE}")"/config.yaml)
function parse_yaml2 {
    local file=$1
    local key=$2
    grep ^$key $file | sed s/#.*//g | grep $key | awk -F':' -vOFS=':' '{$1=""; print $0;}' | awk '{print $2;}' | sed 's/ //g; s/"//g'
}

function modify_yaml {
    local file=$1
    local key=$2
    local value=$3
    sed -i "s|^${key}: .*$|${key}: ${value}|" ${file}
}

