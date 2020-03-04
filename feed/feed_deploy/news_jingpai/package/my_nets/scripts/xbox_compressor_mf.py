#!/usr/bin/python
"""
xbox model compressor
"""

import sys
import math
import time
import re

#WISE
#SHOW_COMPRESS_RATIO : 8192
#CLICK_COMPRESS_RATIO : 8192
#LR_COMPRESS_RATIO : 1048576
#MIO_COMPRESS_RATIO:8192

#PC
#MIO_COMPRESS_RATIO : 1024
#SHOW_COMPRESS_RATIO : 128
#CLICK_COMPRESS_RATIO : 1024
#LR_COMPRESS_RATIO : 8192

#STAMP_COL = 2
SHOW_COL = 3
CLICK_COL = 4
LR_W_COL = 5
LR_G2SUM_COL = 6
FM_COL = 9

#DAY_SPAN = 300

#show clk lr = float
SHOW_RATIO = 1
#SHOW_RATIO = 1024
CLK_RATIO = 8
#CLK_RATIO = 1024
LR_RATIO = 1024
MF_RATIO = 1024

base_update_threshold=0.965
base_xbox_clk_cof=1
base_xbox_nonclk_cof=0.2

def as_num(x):
    y='{:.5f}'.format(x)
    return(y)

def compress_show(xx):
    """
    compress show
    """
    preci = SHOW_RATIO

    x = float(xx)
    return str(int(math.floor(x * preci + 0.5)))


def compress_clk(xx):
    """
    compress clk
    """
    preci = CLK_RATIO

    x = float(xx)
    clk = int(math.floor(x * preci + 0.5))
    if clk == 0:
        return ""
    return str(clk)


def compress_lr(xx):
    """
    compress lr
    """
    preci = LR_RATIO

    x = float(xx)
    lr = int(math.floor(x * preci + 0.5))
    if lr == 0:
        return ""
    return str(lr)

def compress_mf(xx):
    """
    compress mf
    """
    preci = MF_RATIO

    x = float(xx)
    return int(math.floor(x * preci + 0.5))


def show_clk_score(show, clk):
    """
    calculate show_clk score
    """
    return (show - clk) * 0.2 + clk


for l in sys.stdin:
    cols = re.split(r'\s+', l.strip())
    key = cols[0].strip()

    #day = int(cols[STAMP_COL].strip())
    #cur_day = int(time.time()/3600/24)
    #if (day + DAY_SPAN) <= cur_day:
    #    continue

    # cvm features
    show = cols[SHOW_COL]
    click = cols[CLICK_COL]
    pred = ""

    f_show = float(show)
    f_clk = float(click)
    """
    if f_show != 0:
        show_log = math.log(f_show)
    else:
        show_log = 0

    if f_clk != 0:
        click_log =  math.log(f_clk) - show_log
    else:
        click_log = 0
    """
    show_log = f_show
    click_log = f_clk
    #print f_show, f_clk
    #if show_clk_score(f_show, f_clk) < base_update_threshold:
    #    continue

    #show = compress_show(show)
    show = compress_show(show_log)
    #clk = compress_clk(click)
    clk = compress_clk(click_log)

    # personal lr weight
    lr_w = cols[LR_W_COL].strip()
    lr_wei = compress_lr(lr_w)

    # fm weight
    fm_wei = []
    fm_sum = 0
    if len(cols) > 7:
    #fm_dim = int(cols[FM_COL].strip())
    #if fm_dim != 0:
        for v in xrange(FM_COL, len(cols), 1):
            mf_v = compress_mf(cols[v])
            #print mf_v
            fm_wei.append(str(mf_v))
            fm_sum += (mf_v * mf_v)

    sys.stdout.write("%s\t%s\t%s\t%s" % (key, show, clk, pred))
    sys.stdout.write("\t")
    sys.stdout.write("%s" % lr_wei)
    if len(fm_wei) > 0 and fm_sum > 0:
        sys.stdout.write("\t%s" % "\t".join(fm_wei))
    else:
        sys.stdout.write("\t[\t]")
    sys.stdout.write("\n")

