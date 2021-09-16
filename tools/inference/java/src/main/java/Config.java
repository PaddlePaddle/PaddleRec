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
import java.nio.file.Paths;
import java.lang.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.commons.cli.*;

public class Config {
    public static int threadNum;
    public static int batchSize;
    public static float cpuUsageRatio;
    public static int iteration;
    public static String outPerformanceFile;
	public static String inputdata;
	public static String modelFile;

    public Config() {};

    public static void ReadConfig(String[] args) {
        System.out.println(Arrays.asList(args));
		Option opt1 = new Option("t", "threadNum", true, "threrad num");
		opt1.setRequired(true);
		Option opt2 = new Option("bsz", "batchSize", true, "batch size");
		opt2.setRequired(true);
		Option opt3 = new Option("cr", "cpuRatio", true, "cpu usage ratio");
		opt3.setRequired(true);
        Option opt4 = new Option("it", "iteration", true, "iteration num");
        opt4.setRequired(true);
        Option opt5 = new Option("op", "outPerformanceFile", true, "perfomance file");
        opt5.setRequired(true);
		Option opt6 = new Option("inputdata", "inputdata", true, "training file");
        opt6.setRequired(true);
		Option opt7 = new Option("modelFile", "modelFile", true, "model file");
        opt7.setRequired(true);

		Options options = new Options();
		options.addOption(opt1);
		options.addOption(opt2);
		options.addOption(opt3);
        options.addOption(opt4);
        options.addOption(opt5);
		options.addOption(opt6);
		options.addOption(opt7);
        
		CommandLine cli = null;
		CommandLineParser cliParser = new DefaultParser();
		HelpFormatter helpFormatter = new HelpFormatter();
		try {
			cli = cliParser.parse(options, args);
		} catch (ParseException e) {
			helpFormatter.printHelp(">>>>>> test cli options", options);
			e.printStackTrace();
		}

		if (cli.hasOption("t")) {
			threadNum = Integer.parseInt(cli.getOptionValue("t", "1"));
			System.out.println(String.format(">>>>>> thread num: %s", threadNum));
		}
		if (cli.hasOption("bsz")) {
			batchSize = Integer.parseInt(cli.getOptionValue("bsz", "1"));
			System.out.println(String.format(">>>>>> batch size: %s", batchSize));
		}
        if (cli.hasOption("cr")) {
            cpuUsageRatio = Float.parseFloat(cli.getOptionValue("cr", "1.0"));
            System.out.println(String.format(">>>>>> cpu usage ratio: %s", cpuUsageRatio));
        }
        if (cli.hasOption("it")) {
            iteration = Integer.parseInt(cli.getOptionValue("it", "1")); 
			System.out.println(String.format(">>>>>> iteration num: %s", iteration));
        }
        if (cli.hasOption("op")) {
            outPerformanceFile = cli.getOptionValue("op", "performance.txt"); 
			System.out.println(String.format(">>>>>> out performance file: %s", outPerformanceFile));
        }
		if (cli.hasOption("inputdata")) {
			inputdata = cli.getOptionValue("inputdata", "");
			System.out.println(String.format(">>>>>> inputdata: %s", inputdata));
		}
		if (cli.hasOption("modelFile")) {
			modelFile = cli.getOptionValue("modelFile", "");
			System.out.println(String.format(">>>>>> modelFile: %s", modelFile));
		}
    }
}
