with open("session_slot", "r") as fin:
    res = []
    for i in fin:
        res.append("\"" + i.strip() + "\"")
    print ", ".join(res)
