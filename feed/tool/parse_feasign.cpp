#include <stdlib.h>
#include <stdio.h>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

// 将extractor产出的feasign 转 paddle instance
int main(int argc, char * argv[]) {
  ifstream fin(argv[1]);
  int slot_idx = 0;
  unordered_map<int, int> slot_map;
  int slot = 0;
  while (fin >> slot) {
    slot_map[slot] = slot_idx++;
  }
  int slot_num = slot_map.size();
  int max_feasign_num = 10000;
  vector<vector<unsigned long> > slots;
  for (int i = 0; i < slot_num; ++i) {
    vector<unsigned long> tmp;
    tmp.reserve(max_feasign_num);
    slots.push_back(tmp);
  }

  char * linebuf = (char *)calloc(1024*1024*40, sizeof(char));
  if (NULL == linebuf) {
    fprintf(stderr, "memory not enough, exit\n");
    exit(-1);
  }

  int click = 0;
  int show = 0;
  unsigned long feasign = 0;
  int i = 0;
  while (fgets(linebuf, 1024*1024*40, stdin)) {
    char* head_ptr = linebuf;
    for (i = 0; *(head_ptr + i) != ' '; ++i) ;
    head_ptr += i + 1;
    show = strtoul(head_ptr, &head_ptr, 10);
    click = strtoul(head_ptr, &head_ptr, 10);
    int feasign_num = 0;
    while (head_ptr != NULL) {
      feasign = strtoul(head_ptr, &head_ptr, 10);
      if (head_ptr != NULL && *head_ptr == ':') {
        head_ptr++;
        slot = strtoul(head_ptr, &head_ptr, 10);
        feasign_num++;
        if (slot_map.find(slot) == slot_map.end()) {
          continue;
        }
        slots[slot_map[slot]].push_back(feasign);
      } else {
        break;
      }
    }

    int tag = 0;
    float weight = 1;
    bool has_tag = false;
    bool has_weight = false;
    for (int j = 0; *(head_ptr + j) != '\0'; ++j) {
      if (*(head_ptr + j) == '$') {
        has_tag = true;
      } else if (*(head_ptr + j) == '*') {
        has_weight = true;
      }
    }

    if (has_tag) {
        for (i = 0; *(head_ptr + i) != '\0' && *(head_ptr + i) != '$'; ++i) ;
        if (head_ptr + i != '\0') {
          head_ptr += i + 1;
          if (*head_ptr == 'D') {
            tag = 0;
            head_ptr += 1;
          } else {
            tag = strtoul(head_ptr, &head_ptr, 10);
          }
        }
    }

    if (has_weight) {
        for (i = 0; *(head_ptr + i) != '\0' && *(head_ptr + i) != '*'; ++i) ;
        if (head_ptr + i != '\0') {
          head_ptr += i + 1;
          weight = strtod(head_ptr, &head_ptr);
        }
    }

    fprintf(stdout, "1 %d 1 %d", show, click);
    for (size_t i = 0; i < slots.size() - 2; ++i) {
      if (slots[i].size() == 0) {
        fprintf(stdout, " 1 0");
      } else {
        fprintf(stdout, " %lu", slots[i].size());
        for (size_t j = 0; j < slots[i].size(); ++j) {
          fprintf(stdout, " %lu", slots[i][j]);
        }
      }
      slots[i].clear();
      slots[i].reserve(max_feasign_num);
    }
    if (weight == 1.0) {
      fprintf(stdout, " 1 %d 1 %d\n", int(weight), tag);
    } else {
      fprintf(stdout, " 1 %f 1 %d\n", weight, tag);
    }
  }
}
