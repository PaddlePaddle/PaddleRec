// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

vector<string> split (string s, string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

int tokens = 40;
vector<int> cont_min{0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
//vector<int> cont_max{20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50};
vector<int> cont_diff{20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50};
int hash_dim = 1000001;


int main(int argc, char * argv[]) {

    string str;
    //cout.precision(17);
    while (getline(cin, str)) {
        auto line = split(str, "\t");
        if (line.size() != tokens) {
            continue;
        }
        // dense features
        // maybe more complicate data processing
        cout<< "13 ";
        for (int i = 1; i <= 13; ++i) {
            if (line[i] == "") {
                cout<< 0;
            } else {
                cout<< (stod(line[i]) - cont_min[i-1]) / cont_diff[i-1];
            }
            cout<< " ";
        }
        // sparse features
        for (int i = 14; i <= 39; ++i) {
            cout<< "1 ";
            // !!!!!!!!!!!!!!!!!!!!!!! Attention
            // this hash function DO NOT equivalent to the python version
            cout<< hash<string>{}(line[i]) % hash_dim;
            cout<< " ";
        }
        // label
        cout<< "1 ";
        cout<< line[0];
        cout<< endl;

    }

}
