#! /bin/bash
function print(){
    if [ $1 -lt 10 ];
    then
        echo "0"$1
    else
        echo $1
    fi
}
for y in 20190720 20190721 20190722
do
    for((i=0;i<24;i++))
    do
        for((j=0;j<60;j+=5))
        do
            a=`print $i`
            b=`print $j`
            echo $y/$a$b
            mkdir -p data2/$y/$a$b
            for ((k=0;k<1;k++));
            do
                cp ./data/demo_10 data2/$y/$a$b/demo_$k
                # touch data2/$y/$a$b/data.done
               # hadoop fs -Dhadoop.job.ugi=paddle,dltp_paddle@123 -fs afs://yinglong.afs.baidu.com:9902 -put ./data/demo_10 /user/paddle/wangxianming/feed/$y/$a$b/demo_$k
            done
            #hadoop fs -Dhadoop.job.ugi=paddle,dltp_paddle@123 -fs afs://yinglong.afs.baidu.com:9902 -mkdir /user/paddle/wangxianming/feed/$y/$a$b
        done
    done
done
