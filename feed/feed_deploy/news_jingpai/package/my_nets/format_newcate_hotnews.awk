#!/bin/awk -f
{
    if ($1 !~ /^([0-9a-zA-Z])+$/ || $2 !~ /^([0-9])+$/ || $3 !~ /^([0-9])+$/) {
        next;
    }
    show = $2;
    clk = $3;
    if (clk > show) {
        clk = show;
    }
    for (i = 0; i < clk; i++) {
        $2 = "1";
        $3 = "1";
        print $0;
    }
    for (i = 0; i < show - clk; i++) {
        $2 = "1";
        $3 = "0";
        print $0;
    }
}
