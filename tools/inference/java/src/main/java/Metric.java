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

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.paddlepaddle.jni.JniUtils;
import ai.djl.paddlepaddle.engine.PpNDArray;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.BlockingQueue;

import java.io.IOException;
import java.io.*;
import java.nio.file.Paths;
import java.lang.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.cli.*;

public class Metric {
    public double qps;
    public double latency;
    public float cpuUsageRatio;
    public String memUsageInfo;
    public String threadName;
    public NDList batchResult;
    public String outPerformanceFile;

    public long samplecnt;
    
    public Metric () {};

    public void WritePerformance(String outPerformanceFile) {
        try {
            this.outPerformanceFile = outPerformanceFile;
            BufferedWriter out = new BufferedWriter(new FileWriter(outPerformanceFile, true));
            out.write("thread name: " + threadName + "\n");
            out.write("sampleCnts: " + samplecnt + "  total threadNum: " + Config.threadNum + "  batchSize: " + 
                Config.batchSize + "\n");
            out.write("qps: " + qps + "\n");
            out.write("latency: " + latency + "\n");
            out.write("cpu usage ratio: " + cpuUsageRatio + "\n");
            out.write("memory usage info:\n" + memUsageInfo + "\n");
            //out.write("batch result: \n");
            //out.write(batchResult.get(0) + "\n");
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println(String.format("%s: outPerformanceFile created sucess!", threadName));
        System.out.println("\n\n");
    } 

    public static void WriteLog() {
		try {
            BufferedWriter out = new BufferedWriter(new FileWriter("log.txt", true));
			out.write("total batch num: " + ParserInputData.BATCH_NUM + "\n");
			out.write("samples num per batch: " + ParserInputData.BATCH_SIZE + "\n");
			out.write("slots num per sample: " + ParserInputData.SLOT_NUM +  "\n");

            System.out.println("fesign to feature id: ");
			for (String s : ParserInputData.feasignMap.keySet()) {
				out.write(s + ": " + ParserInputData.feasignMap.get(s) + "\n");
			}

			BatchSample batchSample = ParserInputData.batchSamples[0];
			out.write("data in batch 0" + "\n");
			for (Integer slotId : batchSample.features.keySet()) {
				out.write("slot id: " + slotId + "\n");
				for (int i = 0; i < batchSample.features.get(slotId).size(); i++) {
					for (int j = 0; j < batchSample.features.get(slotId).get(i).size(); j++) {
						out.write(batchSample.features.get(slotId).get(i).get(j) + " ");
					}
					out.write("\n");
				}
			}
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        System.out.println("logFile created sucess!");
        System.out.println("\n\n");
    } 
}
