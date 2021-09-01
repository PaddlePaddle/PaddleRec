package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	pd "github.com/paddlepaddle/paddle/paddle/fluid/inference/goapi"
)

type Record struct {
	ID      int64
	Input   []byte // origin line text
	Feasigs map[string][]int64
	Output0 float32
}

func (r *Record) Parse() {
	tokens := bytes.Split(r.Input, []byte(" "))
	feasign := make(map[string][]int64)
	for _, token := range tokens {
		feaslot := bytes.Split(token, []byte(":"))
		if len(feaslot) == 2 {
			fea, _ := strconv.ParseInt(string(feaslot[0]), 10, 64)
			slot := string(feaslot[1])
			if _, ok := feasign[slot]; ok {
				feasign[slot] = append(feasign[slot], fea%400) // feasig to id, % demo only
			} else {
				feasign[slot] = []int64{fea % 400} // feasig to id, % demo only
			}
		} else {
			log.Println("feasign error", string(token), r.ID)
		}
	}
	r.Feasigs = feasign
	//r.Input = nil
}

func numElements(shape []int32) int32 {
	n := int32(1)
	for _, v := range shape {
		n *= v
	}
	return n
}

func feed(outQ chan *Record, filename string, offset int64) {
	start := time.Now()
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	c := 0
	for scanner.Scan() {
		r := Record{
			ID:    int64(c),
			Input: scanner.Bytes(),
		}
		r.Parse()
		outQ <- &r
		c++
	}
	tim := time.Since(start)
	log.Println(fmt.Sprintf("feed %d record in %s, average %f records/s", c, tim, float64(c)/tim.Seconds()))
}

func predict(inQ chan *Record, outQ chan *Record, predictor *pd.Predictor, batchSize int) {
	inNames := predictor.GetInputNames()
	inHandle := make([]*pd.Tensor, len(inNames))
	for i, n := range inNames {
		inHandle[i] = predictor.GetInputHandle(n)
	}

	outNames := predictor.GetOutputNames()
	outHandle := make([]*pd.Tensor, len(outNames))
	for i, n := range outNames {
		outHandle[i] = predictor.GetOutputHandle(n)
	}

	rds := make([]*Record, batchSize)
	for {
		for i := 0; i < batchSize; i++ {
			r, ok := <-inQ
			if !ok {
				log.Println("predict done")
				return
			}
			rds[i] = r
		}

		for i, n := range inNames {
			data := make([]int64, 0, batchSize)
			lod := make([][]uint, 1)
			lod[0] = make([]uint, batchSize+1)
			for j, r := range rds {
				if fea, ok := r.Feasigs[n]; ok {
					data = append(data, fea...)
					lod[0][j+1] = lod[0][len(lod[0])-1] + uint(len(fea))
				} else {
					data = append(data, 0)
					lod[0][j+1] = lod[0][len(lod[0])-1] + 1
				}
			}

			inHandle[i].Reshape([]int32{int32(len(data)), 1})
			inHandle[i].CopyFromCpu(data)
			inHandle[i].SetLod(lod)
		}

		predictor.Run()

		outData := make([]float32, numElements(outHandle[0].Shape()))
		outHandle[0].CopyToCpu(outData)

		for i, o := range outData {
			rds[i].Output0 = o
			//rds[i].Feasigs  = nil
			outQ <- rds[i]
		}
	}
}

func save(inQ chan *Record, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	w := bufio.NewWriter(file)
	for {
		r, ok := <-inQ
		if !ok {
			log.Println("save done")
			return
		}
		w.WriteString(fmt.Sprintf("%d %f \n", r.ID, r.Output0))
		w.Flush()
	}
}

func main() {
	var model string
	var params string
	var thread int
	var cpuMath int
	var batchSize int
	var inputFile string
	var outputFile string
	var feedBuffer int
	var retBuffer int
	var totalTest int
	flag.StringVar(&model, "model", "inference.pdmodel", "the model path")
	flag.StringVar(&params, "params", "inference.pdiparams", "the params path")
	flag.IntVar(&thread, "thread", 1, "thread_num")
	flag.IntVar(&cpuMath, "cpu_math", 4, "cpu_math")
	flag.IntVar(&batchSize, "batch_size", 1, "batch size")
	flag.StringVar(&inputFile, "input_file", "input.txt", "input file")
	flag.StringVar(&outputFile, "output_file", "", "out file")
	flag.IntVar(&feedBuffer, "feed_buffer", 13000, "feed buffer size")
	flag.IntVar(&retBuffer, "ret_buffer", 13000, "ret buffer size")
	flag.IntVar(&totalTest, "total", 10000, "total record stop")

	flag.Parse()

	config := pd.NewConfig()
	config.SetModel(model, params)
	config.SetCpuMathLibraryNumThreads(cpuMath)
	config.SwitchIrOptim(true)
	config.DisableGlogInfo()

	files := strings.Split(inputFile, ",")

	feedQ := make(chan *Record, feedBuffer)
	retQ := make(chan *Record, retBuffer)

	log.Println(fmt.Sprintf("run with thread %d batchsize %d", thread, batchSize))

	for _, file := range files {
		go feed(feedQ, file, 0)
	}
	if outputFile != "" {
		go save(retQ, outputFile)
	}

	// wait feed go on
	time.Sleep(8 * time.Second)

	start := time.Now()

	predictor := pd.NewPredictor(config)
	go predict(feedQ, retQ, predictor, batchSize)
	for i := 1; i < thread; i++ {
		go predict(feedQ, retQ, predictor.Clone(), batchSize)
	}

	c := 0
	for {
		_ = <-retQ
		c++
		if c >= totalTest {
			tim := time.Since(start)
			log.Println(fmt.Sprintf("predict %d record in %s, qps %f records/s", c, tim, float64(c)/tim.Seconds()))
			fmt.Println("RET ", thread, batchSize, c, tim, float64(c)/tim.Seconds(), float64(tim.Nanoseconds())/(float64(c*1000000)))
			break
		}
	}
	close(feedQ)
	close(retQ)
	time.Sleep(5)

}
