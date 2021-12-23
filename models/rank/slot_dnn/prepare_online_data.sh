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
        for((j=0;j<60;j+=30))
        do
            a=`print $i`
            b=`print $j`
            echo $y/$a$b
            mkdir -p data/$y/$a$b
            for ((k=0;k<1;k++));
            do
                cp ./data/demo_10 data/$y/$a$b/demo_$k
                # touch data/$y/$a$b/data.done
            done
        done
    done
done
