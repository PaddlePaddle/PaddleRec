import sys
import os
import paddle
import re
import collections
import time
#import paddle.fluid.incubate.data_generator as dg
import data_generate_base as dg

class MyDataset(dg.MultiSlotDataGenerator):
    def load_resource(self, dictf):
        self._all_slots_dict = collections.OrderedDict()
        with open(dictf, 'r') as f:
            slots = f.readlines()
        for index, slot in enumerate(slots):
            #self._all_slots_dict[slot.strip()] = [False, index + 3] #+3 #
            self._all_slots_dict[slot.strip()] = [False, index + 2]

    def generate_sample(self, line):
        def data_iter_str():
            s = line.split('\t')[0].split()#[1:]
            lineid = s[0]
            elements = s[1:] #line.split('\t')[0].split()[1:]
            padding = "0"
           # output = [("lineid", [lineid]), ("show", [elements[0]]), ("click", [elements[1]])]
            output = [("show", [elements[0]]), ("click", [elements[1]])]
            output.extend([(slot, []) for slot in self._all_slots_dict])
            for elem in elements[2:]:
                if elem.startswith("*"):                                           
                    feasign = elem[1:]                                             
                    slot = "12345"                                                 
                elif elem.startswith("$"):                                         
                    feasign = elem[1:]                                             
                    if feasign == "D":                                             
                        feasign = "0"                                                
                    slot = "23456"                                                 
                else:                                                              
                    feasign, slot = elem.split(':')
                #feasign, slot = elem.split(':')
                if not self._all_slots_dict.has_key(slot):
                    continue
                self._all_slots_dict[slot][0] = True
                index = self._all_slots_dict[slot][1]
                output[index][1].append(feasign)
            for slot in self._all_slots_dict:
                visit, index = self._all_slots_dict[slot]
                if visit:
                    self._all_slots_dict[slot][0] = False
                else:
                    output[index][1].append(padding)
            #print output
            yield output

        return data_iter_str

        def data_iter():
            elements = line.split('\t')[0].split()[1:]
            padding = 0
            output = [("show", [int(elements[0])]), ("click", [int(elements[1])])]
            #output += [(slot, []) for slot in self._all_slots_dict]
            output.extend([(slot, []) for slot in self._all_slots_dict])
            for elem in elements[2:]:
                feasign, slot = elem.split(':')
                if slot == "12345":
                    feasign = float(feasign)
                else:
                    feasign = int(feasign)
                if not self._all_slots_dict.has_key(slot):
                    continue
                self._all_slots_dict[slot][0] = True
                index = self._all_slots_dict[slot][1]
                output[index][1].append(feasign)
            for slot in self._all_slots_dict:
                visit, index = self._all_slots_dict[slot]
                if visit:
                    self._all_slots_dict[slot][0] = False
                else:
                    output[index][1].append(padding)
            yield output
        return data_iter


if __name__ == "__main__":
    #start = time.clock()
    d = MyDataset()
    d.load_resource("all_slot.dict")
    d.run_from_stdin()
    #elapsed = (time.clock() - start)
    #print("Time used:",elapsed)
