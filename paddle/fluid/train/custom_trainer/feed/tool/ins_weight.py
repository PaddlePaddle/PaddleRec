#!/usr/bin/python
import sys
import re
import math

del_text_slot = True
g_ratio = 1
w_ratio = 0.01
slots_str = "6048 6145 6202 6201 6121 6119 6146 6120 6147 6122 6123 6118 6142 6143 6008 6148 6151 6127 6144 6150 6109 6003 6096 6149 6129 6203 6153 6152 6128 6106 6251 7082 7515 7080 6066 7507 6186 6007 7514 6054 6125 7506 10001 6006 6080 7023 6085 10000 6250 6110 6124 6090 6082 6067 7516 6101 6004 6191 6188 6070 6194 6247 6814 7512 10007 6058 6189 6059 7517 10005 7510 7024 7502 7503 6183 7511 6060 6806 7504 6185 6810 6248 10004 6815 6182 10068 6069 6073 6196 6816 7513 6071 6809 6072 6817 6190 7505 6813 6192 6807 6808 6195 6826 6184 6197 6068 6812 7107 6811 6823 6824 6819 6818 6821 6822 6820 6094 6083 6952 6099 6951 6949 6098 7075 6948 6157 6126 7077 6111 6087 6103 6107 6156 6005 6158 7122 6155 7058 6115 7079 7081 6833 6108 6840 6837 7147 7129 6097 6231 6957 7145 6956 7143 6130 7149 7142 6212 6827 7144 6089 6161 7055 6233 6105 7057 6237 6828 6850 6163 7124 6354 6162 7146 6830 7123 6160 6235 7056 6081 6841 6132 6954 6131 6236 6831 6845 6832 6953 6839 6950 7125 7054 6138 6166 6076 6851 6353 7076 7148 6858 6842 6860 7126 6829 6835 7078 6866 6869 6871 7052 6134 6855 6947 6862 6215 6852 7128 6092 6112 6213 6232 6863 6113 6165 6214 6216 6873 6865 6870 6077 6234 6861 6164 6217 7127 6218 6962 7053 7051 6961 6002 6738 6739 10105 7064 6751 6770 7100 6014 6765 6755 10021 10022 6010 10056 6011 6756 10055 6768 10024 6023 10003 6769 10002 6767 6759 10018 6024 6064 6012 6050 10042 6168 6253 10010 10020 6015 6018 10033 10041 10039 10031 10016 6764 7083 7152 7066 6171 7150 7085 6255 10044 10008 7102 6167 6240 6238 6095 10017 10046 6019 6031 6763 6256 6169 6254 10034 7108 7186 6257 10019 6757 10040 6025 7019 7086 10029 10011 7104 6261 6013 6766 10106 7105 7153 7089 6057 7134 7151 7045 7005 7008 7101 6035 7137 10023 6036 6172 7099 7087 6239 7185 6170 10006 6243 6350 7103 7090 7157 6259 7171 6875 7084 7154 6242 6260 7155 7017 7048 7156 6959 7047 10053 7135 6244 7136 10030 7063 6760 7016 7065 7179 6881 7018 6876 10081 10052 10054 10038 6886 10069 7004 10051 7007 7109 10057 6029 6888 10009 6889 7021 10047 6245 6878 10067 6879 6884 7180 7182 10071 7002 6880 6890 6887 10061 6027 6877 6892 10060 6893 7050 10036 7049 10012 10025 7012 7183 10058 7181 10086 6891 6258 6894 6883 7046 6037 7106 10043 10048 10045 10087 6885 10013 10028 7187 10037 10035 10050 6895 7011 7170 7172 10026 10063 10095 10082 10084 6960 10092 10075 6038 7010 7015 10015 10027 10064 7184 10014 10059 7013 7020 10072 10066 10080 6896 10083 10090 6039 10049 7164 7165 10091 10099 6963 7166 10079 10103 7006 7009 7169 6034 7028 7029 7030 7034 7035 7036 7040 7041 7042 10032 6009 6241 7003 7014 7088 13326 13330 13331 13352 13353 6198"
slot_whitelist = slots_str.split(" ")

def calc_ins_weight(params, label):
    """calc ins weight"""
    global g_ratio
    global w_ratio
    slots = []
    s_clk_num = 0
    s_show_num = 0
    active = 0
    attclk_num = 0
    attshow_num = 0
    attclk_avg = 0
    for items in params:
        if len(items) != 2:
            continue
        slot_name = items[0]
        slot_val = items[1]
        if slot_name not in slots:
            slots.append(slot_name)
        if slot_name == "session_click_num":
            s_clk_num = int(slot_val)
        if slot_name == "session_show_num":
            s_show_num = int(slot_val)
        if slot_name == "activity":
            active = float(slot_val) / 10000.0
    w = 1
    # for inactive user 
    if active >= 0 and active < 0.4 and s_show_num >=0 and s_show_num < 20:
        w = math.log(w_ratio * (420 - (active * 50 + 1) * (s_show_num + 1)) + math.e)
        if label == "0":
            w = 1 + (w - 1) * g_ratio
    return w

def filter_whitelist_slot(tmp_line):
    terms = tmp_line.split()
    line = "%s %s %s" % (terms[0], terms[1], terms[2])
    for item in terms[3:]:
        feasign = item.split(':')
        if len(feasign) == 2 and \
            feasign[1] in slot_whitelist:
            line = "%s %s" %(line, item)
    return line

def get_sample_type(line):
    # vertical_type = 20
    # if line.find("13038012583501790:6738") > 0:
    #     return 30
    # vertical_type = 0/5/1/2/9/11/13/16/29/-1
    if (line.find("7408512894065610:6738") > 0) or \
        (line.find("8815887816424655:6738") > 0) or \
        (line.find("7689987878537419:6738") > 0) or \
        (line.find("7971462863009228:6738") > 0) or \
        (line.find("9941787754311891:6738") > 0) or \
        (line.find("10504737723255509:6738") > 0) or \
        (line.find("11067687692199127:6738") > 0) or \
        (line.find("11912112645614554:6738") > 0) or \
        (line.find("15571287443748071:6738") > 0) or \
        (line.find("7127025017546227:6738") > 0): 
        return 20
    return -1

def main():
    """ins adjust"""
    global del_text_slot
    for l in sys.stdin:
        l = l.rstrip("\n")
        items = l.split(" ")
        if len(items) < 3:
            continue
        label = items[2]
        lines = l.split("\t")
        line = lines[0]
        # streaming ins include all ins, sample_type only handle NEWS ins
        sample_type = -1
        if 'NEWS' in l:
            sample_type = get_sample_type(line)
        #line = filter_whitelist_slot(tmp_line)
        if len(lines) >= 4:
            if 'VIDEO' in lines[3]:
                continue
            params = lines[2]
            params = params.split(" ")
            m = [tuple(i.split(":")) for i in params]
            if m is None or len(m) == 0:
                if sample_type > 0:
                    print "%s $%s *1" % (line, sample_type)
                else:
                    print "%s *1" % line
                sys.stdout.flush()
                continue
            weight = calc_ins_weight(m, label)
            if sample_type > 0:
                print "%s $%s *%s" % (line, sample_type, weight)
            else:
                print "%s *%s" % (line, weight)
            sys.stdout.flush()
        else:
            if sample_type > 0:
                print "%s $%s *1" % (line, sample_type)
            else:
                print "%s *1" % line
            sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "0":
            del_text_slot = False
        if len(sys.argv) > 2:
            g_ratio = float(sys.argv[2])
        if len(sys.argv) > 3:
            w_ratio = float(sys.argv[3])
    main()
