using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Core;
using System.IO;

namespace TimTester
{
	class Program
	{
		//=========================================
		//Random
		//=========================================
		static Random rand = new Random(40);
		static Random trainTestrand = new Random(40);

		//=========================================
		//Network
		//=========================================
		static Network network;

		//=========================================
		//Gabor Parameters
		//=========================================
		static float scale = 0.5f;
		static float[] orientations = new float[] { 22.5f, 67.5f, 112.5f, 157.5f };
		static float[] percents = new float[] { 0.15f, 0.12f, 0.10f, 0.07f, 0.05f };
		static float gaborDiv = 4;
		static int gaborRF = 5;
		static int gaborCRF = 7;

		//=========================================
		//Kernel Parameters
		//=========================================
		static float ap = 0.05f;
		static float an = 0.025f;
		static float antiAp = ap;
		static float antiAn = 0.005f;
		static float[] thresholds;
		static int numberOfFeatures = 10;
		static int SRF = 17;
		static int threshold = 42;
		static float pr_interval = 0.8f;
		static float pr_min = 0.2f;
		static float dropOutProbability = 0.3f;

		//=========================================
		//Classification task
		//=========================================
		static int numberOfClasses = 2;
		//NOTE:
		//Those the network should remain silent for, must be added after main classes
		static string[] allImages = new string[]
		{
			@"..\..\dataset\caltech\Face",
			@"..\..\dataset\caltech\Motorbike",

		};
		static int[] startIndices;
		static int numberOfTrainingSamples = 50;
		static string[] classNames =
		{
			"Face",
			"Motorbike",
		};

		static List<int> labels;
		static int[] neuronAssociations = new int[numberOfFeatures];
		
		static GaborLayer gaborLayer;

		static Performance bestTrainPerformance;
		static Performance bestTestPerformance;

		static Kernel bestTrainKernel = null;
		static Kernel bestTestKernel = null;

		static void Main(string[] args)
		{
			string taskName = "CaltechFaceMotorbike";
			int epochs = 200;
			
			//Network
			network = new Network(rand);
			
			gaborLayer = new GaborLayer(allImages,
			scale, orientations, gaborRF, gaborDiv, gaborCRF, percents);
			
			//Labels
			labels = new List<int>();
			startIndices = new int[allImages.Length + 1];
			for(int i = 0; i < allImages.Length; i++)
			{
				string folder = allImages[i];
				int cnt = Directory.EnumerateFiles(folder, "*.png").Count();
				startIndices[i + 1] = startIndices[i] + cnt;
				
				int lbl = i;
				if (i >= numberOfClasses)
					lbl = -1; //for silence
				labels.AddRange(Enumerable.Repeat(lbl, cnt));
			}
			
			//Associations
			int d = numberOfFeatures / numberOfClasses;
			for(int i = 0; i < numberOfFeatures; i++)
			{
				neuronAssociations[i] = i / d;
			}
			
			//Thresholds
			thresholds = Enumerable.Repeat((float)threshold, numberOfFeatures).ToArray();

			int c = 1;
			string tempTaskName = Path.Combine(taskName, c.ToString());
			Directory.CreateDirectory(tempTaskName);
			MyModel(tempTaskName, c, epochs);
		}
		
		static void MyModel(string taskName, int c, int epoch)
		{
			if(!Directory.Exists(taskName))
				Directory.CreateDirectory(taskName);

			SaveModelConfig(taskName);

			StreamWriter trainPerfWriter = new StreamWriter(Path.Combine(taskName, "trainHistory.txt"));
			StreamWriter testPerfWriter = new StreamWriter(Path.Combine(taskName, "testHistory.txt"));
			StreamWriter bestPerfWriter = new StreamWriter(Path.Combine(taskName, "bestPerformance.txt"));
			trainPerfWriter.WriteLine("# Corrects Wrongs Silences");
			testPerfWriter.WriteLine("# Corrects Wrongs Silences");

			//Dividing Test/Train
			List<List<SpikeData>> testOrderedData = new List<List<SpikeData>>();
			List<List<SpikeData[,]>> testData = new List<List<SpikeData[,]>>();
			List<string> testNames = new List<string>();
			List<int> testLabels = new List<int>();

			List<List<SpikeData>> trainOrderedData = new List<List<SpikeData>>();
			List<List<SpikeData[,]>> trainData = new List<List<SpikeData[,]>>();
			List<string> trainNames = new List<string>();
			List<int> trainLabels = new List<int>();
			
			bool[] visit = new bool[gaborLayer.OrderedData.Count];
			int[] trainCnt = Enumerable.Repeat(numberOfTrainingSamples, allImages.Length).ToArray();

			for (int i = 0; i < allImages.Length; i++)
			{
				while (trainCnt[i] > 0)
				{
					int idx = trainTestrand.Next(startIndices[i], startIndices[i + 1]);
					if (visit[idx] == false)
					{
						visit[idx] = true;
						--trainCnt[i];
					}
				}
			}

			for (int j = 0; j < gaborLayer.OrderedData.Count; j++)
			{
				if (!visit[j])
				{
					testOrderedData.Add(gaborLayer.OrderedData[j]);
					testData.Add(gaborLayer.SpikeData2D[j]);
					testNames.Add(gaborLayer.FileNames[j]);
					testLabels.Add(labels[j]);
				}
				else
				{
					trainOrderedData.Add(gaborLayer.OrderedData[j]);
					trainData.Add(gaborLayer.SpikeData2D[j]);
					trainNames.Add(gaborLayer.FileNames[j]);
					trainLabels.Add(labels[j]);
				}
			}

			
			//TODO: Loading kernel
			Kernel kernel = new Kernel("S2RL", numberOfFeatures, orientations.Length,
				SRF, thresholds, 1, ap, an, antiAp, antiAn, 0.8f, 0.05f, rand);
			kernel.Ap = ((1f - 1.0f / numberOfClasses) * pr_interval + pr_min) * ap;
			kernel.An = ((1f - 1.0f / numberOfClasses) * pr_interval + pr_min) * an;
			kernel.AntiAp = ((1.0f / numberOfClasses) * pr_interval + pr_min) * antiAp;
			kernel.AntiAn = ((1.0f / numberOfClasses) * pr_interval + pr_min) * antiAn;
			
			int maxIteration = epoch;
			for (int iter = 1; iter <= maxIteration; iter++)
			{
				Console.WriteLine(iter);

				//Train
				bool[] active = Enumerable.Repeat(true, numberOfFeatures).Select(a => rand.NextDouble() >= dropOutProbability).ToArray();
				network.TrainKernelRL(trainOrderedData, trainData, kernel, 1, 1, trainLabels, neuronAssociations, active);

				//Compute Train/Test Performance
				List<List<int>> trainResponses = null;
				List<List<int>> testResponses = null;
				trainResponses =
				network.GetKernelFirstSpikes(trainOrderedData, trainData, kernel);
				testResponses =
				network.GetKernelFirstSpikes(testOrderedData, testData, kernel);

				Console.WriteLine();
				int correct = 0, wrong = 0, silence = 0;
				for (int i = 0; i < trainOrderedData.Count; ++i)
				{
					int idx = trainResponses[i].FindIndex(tr => tr == 1);
					if (idx == -1)
					{
						if (trainLabels[i] != -1)
							++silence;
						else
							++correct;
					}
					else if (neuronAssociations[idx] == trainLabels[i])
						++correct;
					else
						++wrong;
				}

				Performance tempTrainPerf = new Performance(iter, correct, wrong, silence);
				if (bestTrainPerformance == null || bestTrainPerformance.Corrects <= tempTrainPerf.Corrects)
				{
					bestTrainPerformance = tempTrainPerf;
					bestTrainKernel = Kernel.GetCopy(kernel, kernel.Name + "Train" + iter);
				}
				trainPerfWriter.WriteLine($"{tempTrainPerf.Corrects} {tempTrainPerf.Wrongs} {tempTrainPerf.Silences}");
				Console.WriteLine("Best Train: " + bestTrainPerformance);

				kernel.Ap = (tempTrainPerf.Wrongs * pr_interval + pr_min) * ap;
				kernel.An = (tempTrainPerf.Wrongs * pr_interval + pr_min) * an;
				kernel.AntiAp = (tempTrainPerf.Corrects * pr_interval + pr_min) * antiAp;
				kernel.AntiAn = (tempTrainPerf.Corrects * pr_interval + pr_min) * antiAn;

				correct = 0; wrong = 0; silence = 0;
				for (int i = 0; i < testOrderedData.Count; ++i)
				{
					int idx = testResponses[i].FindIndex(tr => tr == 1);
					if (idx == -1)
					{
						if (testLabels[i] != -1)
							++silence;
						else
							++correct;
					}
					else if (neuronAssociations[idx] == testLabels[i])
						++correct;
					else
						++wrong;
				}
				Performance tempTestPerf = new Performance(iter, correct, wrong, silence);
				if (bestTestPerformance == null || bestTestPerformance.Corrects <= tempTestPerf.Corrects)
				{
					bestTestPerformance = tempTestPerf;
					bestTestKernel = Kernel.GetCopy(kernel, kernel.Name + "Test" + iter);
					if (wrong == 0 && silence == 0)
					{
						break;
					}
				}
				testPerfWriter.WriteLine($"{tempTestPerf.Corrects} {tempTestPerf.Wrongs} {tempTestPerf.Silences}");
				Console.WriteLine("Best Test: " + bestTestPerformance);
				Console.WriteLine($"Params: ap: {kernel.Ap}, an: {kernel.An}, antiap: {kernel.AntiAp}, antian: {kernel.AntiAn}");
				Console.WriteLine();

				if (iter % 25 == 0)
				{
					trainPerfWriter.Flush();
					testPerfWriter.Flush();
				}

				if (iter % 250 == 0)
				{
					bestPerfWriter.WriteLine($"#Iteration {iter}");
					bestPerfWriter.WriteLine("#Train");
					bestPerfWriter.WriteLine(bestTrainPerformance);
					bestPerfWriter.WriteLine("#Test");
					bestPerfWriter.WriteLine(bestTestPerformance);
					bestPerfWriter.Flush();
				}
			}

			trainPerfWriter.Close();
			testPerfWriter.Close();

			bestPerfWriter.WriteLine("#Train");
			bestPerfWriter.WriteLine(bestTrainPerformance);
			bestPerfWriter.WriteLine("#Test");
			bestPerfWriter.WriteLine(bestTestPerformance);
			bestPerfWriter.Close();

			network.SaveKernel(taskName, bestTrainKernel, gaborLayer.gabor.gaborFeatures);
			network.SaveKernel(taskName, bestTestKernel, gaborLayer.gabor.gaborFeatures);

			MyModelTest(Path.Combine(taskName, "testOnTest"), bestTestKernel,
				testOrderedData,
				testData,
				testLabels,
				testNames.Select(s => Path.GetFileNameWithoutExtension(s)).ToArray());

			MyModelTest(Path.Combine(taskName, "testOnTrain"), bestTestKernel,
				trainOrderedData,
				trainData,
				trainLabels,
				trainNames.Select(s => Path.GetFileNameWithoutExtension(s)).ToArray());
		}

		static void SaveModelConfig(string taskName)
		{
			StreamWriter configWriter = new StreamWriter(Path.Combine(taskName, "config.txt"));

			configWriter.WriteLine($"gabor.scales {scale}");
			configWriter.WriteLine($"gabor.orientations {string.Join(",", orientations)}");
			configWriter.WriteLine($"gabor.inhibition {string.Join(",", percents)}");
			configWriter.WriteLine($"gabor.div {gaborDiv}");
			configWriter.WriteLine($"gabor.rf {gaborRF}");
			configWriter.WriteLine($"gabor.crf {gaborCRF}");

			configWriter.WriteLine($"kernel.ap {ap}");
			configWriter.WriteLine($"kernel.an {an}");
			configWriter.WriteLine($"kernel.antiap {antiAp}");
			configWriter.WriteLine($"kernel.antian {antiAn}");
			configWriter.WriteLine($"kernel.thresholds {string.Join(",", thresholds)}");
			configWriter.WriteLine($"kernel.numberoffeatures {numberOfFeatures}");
			configWriter.WriteLine($"kernel.srf {SRF}");

			configWriter.WriteLine($"printerval {pr_interval}");
			configWriter.WriteLine($"prmin {pr_min}");

			configWriter.Close();
		}

		static void MyModelTest(string taskName, Kernel kernel, List<List<SpikeData>> orderedSpikes,
			List<List<SpikeData[,]>> spikeData, List<int> labels, string[] inputNames)
		{
			if (!Directory.Exists(taskName))
				Directory.CreateDirectory(taskName);

			//Finding greatest dim values
			int maxHeight = 0;
			int maxWidth = 0;
			foreach (var level2 in spikeData)
			{
				foreach (var level3 in level2)
				{
					if (level3.GetLength(0) > maxHeight)
						maxHeight = level3.GetLength(0);
					if (level3.GetLength(1) > maxWidth)
						maxWidth = level3.GetLength(1);
				}
			}

			int[] correctResponsCnt = new int[numberOfFeatures];
			int[] averages = new int[numberOfClasses];
			int[] wrongResponsCnt = new int[numberOfFeatures];
			int[,] confusionMatrix = new int[numberOfClasses, numberOfClasses + 1];
			int maxInConfusion = 0;
			StreamWriter missWriter = new StreamWriter(Path.Combine(taskName, "misses.txt"));

			List<List<int>> output = network.GetKernelFirstSpikes(orderedSpikes, spikeData, kernel);
			int corrects = 0, wrongs = 0, silences = 0;
			for (int i = 0; i < output.Count; i++)
			{
				if (labels[i] == -1)
					continue;
				int winner = output[i].IndexOf(1);
				if (winner == -1)
				{
					++confusionMatrix[labels[i], numberOfClasses];
					if (confusionMatrix[labels[i], numberOfClasses] > maxInConfusion)
						maxInConfusion = confusionMatrix[labels[i], numberOfClasses];
					++silences;
				}
				else if (neuronAssociations[winner] != labels[i])
				{
					++confusionMatrix[labels[i], neuronAssociations[winner]];
					if (confusionMatrix[labels[i], neuronAssociations[winner]] > maxInConfusion)
						maxInConfusion = confusionMatrix[labels[i], neuronAssociations[winner]];
					missWriter.WriteLine($"{inputNames[i]}({labels[i]}) -> Neuron:{winner} Class:{neuronAssociations[winner]}");
					++wrongResponsCnt[winner];
					++averages[neuronAssociations[winner]];
					++wrongs;
				}
				else
				{
					++confusionMatrix[labels[i], neuronAssociations[winner]];
					if (confusionMatrix[labels[i], neuronAssociations[winner]] > maxInConfusion)
						maxInConfusion = confusionMatrix[labels[i], neuronAssociations[winner]];
					++correctResponsCnt[winner];
					++averages[neuronAssociations[winner]];
					++corrects;
				}
			}
			missWriter.Close();

			Performance testPerformance = new Performance(0, corrects, wrongs, silences);
			Console.WriteLine(testPerformance);

			//Involvements
			StreamWriter[] statWriter = new StreamWriter[numberOfClasses];
			for (int c = 0; c < numberOfClasses; c++)
			{
				statWriter[c] = new StreamWriter(Path.Combine(taskName, $"CWStat_c{c}.txt"));
				List<int> neuronsForClass = Enumerable.Range(0, numberOfFeatures).
					Where(i => neuronAssociations[i] == c).ToList();
				averages[c] /= neuronsForClass.Count;
				for (int i = 0; i < neuronsForClass.Count; i++)
				{
					statWriter[c].WriteLine($"{neuronsForClass[i]} {correctResponsCnt[neuronsForClass[i]]} {wrongResponsCnt[neuronsForClass[i]]}");
				}
			}
			foreach (StreamWriter writer in statWriter)
				writer.Close();

			StreamWriter gnuScript = new StreamWriter(Path.Combine(taskName,"involvement.plt"));
			gnuScript.WriteLine($"set xrange[0:*]; set yrange[0:*]");
			gnuScript.WriteLine($"set terminal pdf");
			gnuScript.WriteLine($"set xlabel 'Neuron number'");
			gnuScript.WriteLine($"set ylabel 'Response count over all test images'");
			gnuScript.WriteLine($"set xtics rotate");
			gnuScript.WriteLine($"set grid ytics");
			gnuScript.WriteLine("set boxwidth 0.25 absolute");
			gnuScript.WriteLine("set style fill solid 0.25 border");
			gnuScript.WriteLine("set offsets 0.5, 0.5, 0, 0");
			for (int i = 0; i < numberOfClasses; i++)
			{
				gnuScript.WriteLine($"	t = sprintf('Class{i}: {classNames[i]}')");
				gnuScript.WriteLine($"	set title t");
				gnuScript.WriteLine($"	outfile = sprintf('invovlement_{i}.pdf')");
				gnuScript.WriteLine($"	set output outfile");
				gnuScript.WriteLine($"	infile = sprintf('CWStat_c{i}.txt')");
				gnuScript.WriteLine($"	plot infile u 0:2:(0.75):xticlabels(1) with boxes linecolor rgb '#00a148' title 'Corrects', infile u 0:3:(0.25) with boxes linecolor rgb '#ff0000' title 'Wrongs', {averages[i]} title ''");
				gnuScript.WriteLine();
			}

			gnuScript.Close();

			//Confusion Matrix
			int maxLength = 0;
			while(maxInConfusion > 0)
			{
				maxInConfusion /= 10;
				maxLength++;
			}
			maxLength = Math.Max(maxLength, classNames.Max(s => s.Length));
			maxLength = Math.Max(maxLength, "Silence".Length);
			StreamWriter confusionWriter = new StreamWriter(Path.Combine(taskName, "confusionMatrix.txt"));

			confusionWriter.Write("{0," + maxLength + "}|", " ");
			for (int i = 0; i < classNames.Length; i++)
			{
				confusionWriter.Write("{0," + maxLength + "}|", classNames[i]);
			}
			confusionWriter.WriteLine("{0," + maxLength + "}", "Silence");

			for (int c = 0; c < classNames.Length; c++)
			{
				confusionWriter.Write("{0," + maxLength + "}|", classNames[c]);
				for (int i = 0; i < classNames.Length; i++)
				{
					confusionWriter.Write("{0," + maxLength + "}|", confusionMatrix[c, i]);
				}
				confusionWriter.WriteLine("{0," + maxLength + "}", confusionMatrix[c, classNames.Length]);
			}

			confusionWriter.Close();
		}
	}
}
