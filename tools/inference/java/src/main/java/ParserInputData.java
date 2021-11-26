/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

import java.util.*;
import java.io.IOException;
import java.nio.file.Files;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.LinkedBlockingQueue;


public class ParserInputData {

    public ParserInputData() {}

    public static LinkedBlockingQueue<Integer> queue = new LinkedBlockingQueue<Integer>();
    public static int BATCH_SIZE = 2;
    public static final int BUFFER_MAX = 20480;
    public static int BATCH_NUM;
    public static final int SLOT_NUM = 300;
    public static BatchSample[] batchSamples = new BatchSample[BUFFER_MAX];
    public static TreeMap<String, Integer> feasignMap = new TreeMap<String, Integer>();

    public static void ReadInputData() {
        Integer[] slotIds = new Integer[SLOT_NUM];
        String[] inputVarnames =  new String[SLOT_NUM];
        for (int i = 2; i <= 301; i++) {
            inputVarnames[i - 2] = String.valueOf(i);
            slotIds[i - 2] = i;
        }
        for (int i = 0; i < BUFFER_MAX; i++) {
            batchSamples[i] = new BatchSample();
        }
        try {
            FileInputStream inputStream = new FileInputStream(Config.inputdata);
            BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(inputStream));
            int batchIdx = 0;
            int lineCnt = 0;
            String line = null;
            while((line = bufferedReader.readLine()) != null) {
                //System.out.println(line);
                TreeMap<Integer, ArrayList<Integer>> oneSample = new TreeMap<Integer, ArrayList<Integer>>();
                lineCnt++;
                String[] ele;
                String[] feature;
                String delimeter1 = " ";
                String delimeter2 = ":";
                ele = line.split(delimeter1);
                for (String x : ele) {
                    feature = x.split(delimeter2);
                    if (!feasignMap.containsKey(feature[0])) {
                        feasignMap.put(feature[0], feasignMap.size() + 1);
                    }
                    int feasign = feasignMap.get(feature[0]);
                    int slotId = Integer.parseInt(feature[1]);
                    if (!oneSample.containsKey(slotId)) {
                        ArrayList<Integer> arr = new ArrayList<Integer>();
                        arr.add(feasign);
                        oneSample.put(slotId, arr);
                    } else {
                        oneSample.get(slotId).add(feasign);
                    }
                }
                for (Integer slotId : slotIds) {
                    if (oneSample.containsKey(slotId)) {
                        continue;
                    }
                    ArrayList<Integer> arr = new ArrayList<Integer>();
                    arr.add(0);
                    oneSample.put(slotId, arr);
                }
                for (Integer slotId : slotIds) {
                    if (!batchSamples[batchIdx].features.containsKey(slotId)) {
                        ArrayList<ArrayList<Integer>> arr = new ArrayList<ArrayList<Integer>>();
                        ArrayList<Integer> cnt2 = new ArrayList<Integer>();
                        arr.add(oneSample.get(slotId));
                        batchSamples[batchIdx].features.put(slotId, arr);
                        cnt2.add(oneSample.get(slotId).size());
                        batchSamples[batchIdx].featureCnts.put(slotId, cnt2);
                    } else {
                        batchSamples[batchIdx].features.get(slotId).add(oneSample.get(slotId));
                        batchSamples[batchIdx].featureCnts.get(slotId).add(oneSample.get(slotId).size());
                    }
                }
                if (lineCnt == BATCH_SIZE) {
                    lineCnt = 0;
                    queue.put(batchIdx);
                    //System.out.println("generate batchIdx: " + batchIdx);
                    batchIdx++;
                }
            }
            BATCH_NUM = batchIdx;
            inputStream.close();
            bufferedReader.close();

        } catch (Exception e) {
            e.printStackTrace();
            return;
        }
    }

    public static void TestParseInputData() {
        System.out.println("total batch num: " + batchSamples.length);
        BatchSample batchSample = batchSamples[0];
        System.out.println("data in batch 0");
        for (Integer slotId : batchSample.features.keySet()) {
            System.out.println("slot id: " + slotId);
            for (int i = 0; i < batchSample.features.get(slotId).size(); i++) {
                for (int j = 0; j < batchSample.features.get(slotId).get(i).size(); j++) {
                    System.out.print(batchSample.features.get(slotId).get(i).get(j) + " ");
                }
                System.out.print("\n");
            }
        }
    }

    public static void TestPrintFeasignMap() {
    	for (String s : feasignMap.keySet()) {
	        System.out.println(s + ": " + feasignMap.get(s));
        }
    }
}
