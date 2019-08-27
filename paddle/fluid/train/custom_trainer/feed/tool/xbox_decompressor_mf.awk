#!/bin/awk -f
{
    OFS="\t";
    SHOW_RATIO = 1;
    CLK_RATIO = 8;
    LR_RATIO = 1024;
    MF_RATIO = 1024;
}

function decompress_show(x) {
    x = x * 1.0 / SHOW_RATIO;
    return x;
}

function decompress_clk(x) {
    if (x == "") {
        x = 0;
    }
    x = x * 1.0 / CLK_RATIO;
    return x;
}

function decompress_lr(x) {
    return x * 1.0 / LR_RATIO;
}

function decompress_mf(x) {
    return x * 1.0 / MF_RATIO;
}

function show_clk_sore(show, clk, nonclk_coeff, clk_coeff) {
    return (show - clk) * nonclk_coeff + clk * clk_coeff;
}

#key, show, clk, pred, lr_w, mf_w or [\t]
{
    l=split($0, a, "\t");

    show = decompress_show(a[2]);
    click = decompress_clk(a[3]);
    lr = decompress_lr(a[5]);
    printf("%s\t0\t0\t%s\t%s\t%s\t0\t", a[1], show, click, lr);
    if (l == 7) {
        printf("0\n");
    } else {
        printf("%d", l-5)
        for(i = 6; i <= l; i++) {
            printf("\t%s", decompress_mf(a[i]));
        }
        printf("\t0\n");
    }
}
