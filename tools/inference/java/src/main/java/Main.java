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

import java.io.IOException;
import java.nio.file.Paths;
import java.lang.*;
import java.util.*;
import java.lang.management.MemoryMXBean;
import java.lang.management.ManagementFactory;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.cli.*;

public class Main {
	private static final Logger logger = LoggerFactory.getLogger(Main.class);
	private static MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();

	//public static long t;
	
	public static void main(String[] args) throws IOException, MalformedModelException, TranslateException, ModelNotFoundException {
		String arch = System.getProperty("os.arch");
		if (!"x86_64".equals(arch) && !"amd64".equals(arch)) {
            logger.warn("{} is not supported.", arch);
            return;
        }
		Config config = new Config();
		config.ReadConfig(args);

		ParserInputData.BATCH_SIZE = config.batchSize;
		ParserInputData.ReadInputData();
		Metric.WriteLog();
		for (int i = 0; i < ParserInputData.BATCH_NUM; i++) {
			listIn.add(GetNDListIn(i));
		}
		//ParserInputData.TestParseInputData();
		long timeInferStart = System.currentTimeMillis();
		Criteria<NDList, NDList> criteria = Criteria.builder()
			.setTypes(NDList.class, NDList.class)
			.optEngine("PaddlePaddle")
			.optModelPath(Paths.get(Config.modelFile))
			.optModelName("rec_inference")
			.optOption("removePass", "repeated_fc_relu_fuse_pass")
			.optDevice(Device.cpu())
			.optProgress(new ProgressBar())
			.build();

		ZooModel<NDList, NDList> model = criteria.loadModel();

		//TestMain();
		try {
			List<InferCallable> callables = new ArrayList<>(Config.threadNum);
			for (int i = 0; i < Config.threadNum; i++) {
				callables.add(new InferCallable(model, i));
			}
			int successThreads = 0;
			List<Future<NDList>> futures = new ArrayList<Future<NDList>>();
			ExecutorService es = Executors.newFixedThreadPool(Config.threadNum);
			for (InferCallable callable : callables) {
				futures.add(es.submit(callable));
			}
			for (Future<NDList> future : futures) {
				if (future.get() != null) {
					++successThreads;
				}
			}
			System.out.println("successfull threads: " + successThreads);
			//System.out.print("read data time cost: " + t);
			long timeInferEnd = System.currentTimeMillis();
			System.out.print("e2e time cost: " + (timeInferEnd - timeInferStart));
			Metric metric = GetMetricInfo(timeInferEnd - timeInferStart, ParserInputData.BATCH_NUM * Config.batchSize, futures.get(0).get());
			metric.WritePerformance(Config.outPerformanceFile);

			for (InferCallable callable : callables) {
				callable.close();
			}
			es.shutdown();
		} catch (InterruptedException | ExecutionException e) {
			logger.error("", e);
		}
	}

	public static class InferCallable implements Callable<NDList> {
		private Predictor<NDList, NDList> predictor;
		private int threadId;
		private NDList batchResult = null;
		private Metric metric = new Metric();
		public InferCallable(ZooModel<NDList, NDList> model, int threadId) {
			this.predictor = model.newPredictor();
			this.threadId = threadId;
		}
		
		public NDList call() {
			long t1 = 0, t2 = 0, t = 0;
			long sampleCnts = 0;
			try {
				while (ParserInputData.queue.size() > 0) {
					int batchIdx = ParserInputData.queue.take();
					NDList batchListIn = GetNDListIn(batchIdx);
					t1 = System.currentTimeMillis();
					sampleCnts += 1;
					batchResult = predictor.predict(batchListIn);
					t2 = System.currentTimeMillis();
					t += (t2 - t1);
				}
				System.out.println("paddle predictor qps: " + 1000.0 * sampleCnts * ParserInputData.BATCH_SIZE / t);
				return batchResult;
			} catch(Exception e) {
				e.printStackTrace();
			}
			return batchResult;
		}

		public void close() {
			predictor.close();
		}
	}

	public static Metric GetMetricInfo(long t, long sampleCnts, NDList batchResult) {
		Metric metric = new Metric();
		metric.threadName = Thread.currentThread().getName();
		metric.cpuUsageRatio = Config.cpuUsageRatio;
		metric.samplecnt = sampleCnts;
		metric.latency = 1.0 * t / metric.samplecnt;
		metric.qps = 1000.0 * metric.samplecnt / t;
		//metric.memUsageInfo = memoryMXBean.getHeapMemoryUsage().toString();
		metric.batchResult = batchResult;
		return metric;
	}

	public static ArrayList<NDList> listIn = new ArrayList<NDList>();
	public static ArrayList<NDList> listOut = new ArrayList<NDList>();

	public static NDList GetNDListIn(int batchIdx) {
		BatchSample batchSample = ParserInputData.batchSamples[batchIdx];
		NDManager manager = NDManager.newBaseManager();
		NDList list = new NDList();
		for (Integer slotId : batchSample.features.keySet()) {
			long[] inputFeasignIds = new long [batchSample.length(slotId)];
			int k = 0;
			long[][] lod = new long[1][ParserInputData.BATCH_SIZE + 1];
			lod[0][0] = 0;
			for (int sampleIdx = 0; sampleIdx < batchSample.features.get(slotId).size(); sampleIdx++) {
				lod[0][sampleIdx + 1] = lod[0][sampleIdx] + batchSample.featureCnts.get(slotId).get(sampleIdx);
				for (int m = 0; m < batchSample.features.get(slotId).get(sampleIdx).size(); m++) {
					inputFeasignIds[k] = batchSample.features.get(slotId).get(sampleIdx).get(m);
				}
			}
			NDArray inputData = manager.create(inputFeasignIds, new Shape(inputFeasignIds.length, 1));
			((PpNDArray)inputData).setLoD(lod);
			list.add(inputData);
		}
		return list;
	}

	public static void TestMain() {
		System.out.println("total batch num: " + ParserInputData.BATCH_NUM);
		System.out.println("samples num per batch: " + ParserInputData.BATCH_SIZE);
		System.out.println("slots num per sample: " + ParserInputData.SLOT_NUM);
		for (int i = 0; i < ParserInputData.BATCH_NUM; i++) {
			System.out.println("NDList In for batch " + i + ": " + listIn.get(0));
			System.out.println("NDList Out for batch " + i + ": " + listOut.get(0));
		}
	}
}
