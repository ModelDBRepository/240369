using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;

namespace Core
{
	//Manages spike propagation through layers
	public class Network
	{
		public List<List<SpikeData>> OrderedSpikes { get; set; }
		public List<List<SpikeData[,]>> Spikes3D { get; set; }
		private Random rnd;
		
		public Network(Random random)
		{
			rnd = random;
		}

		//Train layer S2 with all of the input images
		//This is for the pure STDP (no RL)
		public void TrainKernel(List<List<SpikeData>> orderedSpikes, List<List<SpikeData[,]>> spikeData,
			Kernel kernel, int numberOfEpoch, int numberOfImageRepeat,
			int learningRateIncCount = 400, float learningRateCo = 2.0f)
		{
			OrderedSpikes = orderedSpikes;
			Spikes3D = spikeData;
			
			//training new kernel
			Console.WriteLine($"Training kernel {kernel.Name}...");
			Console.WriteLine($"Shuffling input...");
			int[] shuffled = Enumerable.Range(0, OrderedSpikes.Count).OrderBy(s => rnd.Next()).ToArray();

			int cnt = 1;
			for (int i = 0; i < numberOfEpoch; i++)
			{
				Console.WriteLine("Epoch: " + (i + 1));
				for (int m = 0; m < OrderedSpikes.Count; m++)
				{
					int j = shuffled[m];
					for (int k = 0; k < numberOfImageRepeat; k++, cnt++)
					{
						if (cnt % learningRateIncCount == 0)
						{
							if (kernel.Ap < 0.25)
							{
								Console.WriteLine(cnt);
								kernel.Ap *= learningRateCo;
								kernel.An *= learningRateCo;
							}
						}
						kernel.TrainKernel(OrderedSpikes[j], Spikes3D[j]);
						kernel.ApplySTDP(true);
					}
				}
			}
			Console.WriteLine("Done.");
		}

		//Train layer S2 with all of the input images
		//This is for reinforcement learning
		public void TrainKernelRL(List<List<SpikeData>> orderedSpikes, List<List<SpikeData[,]>> spikeData,
			Kernel kernel, int numberOfEpoch, int numberOfImageRepeat, List<int> labels, int[] neuronAssociations,
			bool[] isActive = null)
		{
			OrderedSpikes = orderedSpikes;
			Spikes3D = spikeData;

			//training new kernel
			Console.WriteLine($"Training kernel {kernel.Name}... (RL)");
			Console.WriteLine($"Shuffling input...");
			int[] shuffled = Enumerable.Range(0, OrderedSpikes.Count).OrderBy(s => rnd.Next()).ToArray();

			int cnt = 1;
			for (int i = 0; i < numberOfEpoch; i++)
			{
				Console.WriteLine("Epoch: " + (i + 1));
				for (int m = 0; m < OrderedSpikes.Count; m++)
				{
					int j = shuffled[m];
					for (int k = 0; k < numberOfImageRepeat; k++, cnt++)
					{
						List<SpikeData> winners = kernel.TrainKernelAndGetSTDPWinners(OrderedSpikes[j], Spikes3D[j], isActive);
						kernel.ApplySTDP(winners.Count != 0 && neuronAssociations[winners[0].Feature] == labels[j]);
					}
				}
			}
			Console.WriteLine("Done.");
		}
		
		//Extracts S2 first spike for each image
		public List<List<int>> GetKernelFirstSpikes(List<List<SpikeData>> orderedSpikes,
			List<List<SpikeData[,]>> spikeData, Kernel kernel)
		{
			List<List<SpikeData>> winners = new List<List<SpikeData>>();
			for (int i = 0; i < orderedSpikes.Count; i++)
			{
				winners.Add(kernel.TestKernelForFirstSpikes(orderedSpikes[i], spikeData[i]));
			}

			Console.Write("Computing first spikes...");
			List<List<int>> firstSpikes = new List<List<int>>(spikeData.Count);
			for (int img = 0; img < orderedSpikes.Count; img++)
			{
				List<int> spikeOrNot = Enumerable.Repeat(0, kernel.NumberOfFeature).ToList();
				if (winners[img].Count != 0)
					spikeOrNot[winners[img][0].Feature] = 1;
				firstSpikes.Add(spikeOrNot);
			}
			Console.WriteLine(" Done.");
			return firstSpikes;
		}

		//Extracts S2 potentials for each image
		private List<List<float[,]>> TestNetworkForPotentials(List<List<SpikeData>> orderedSpikes,
			List<List<SpikeData[,]>> spikeData, Kernel kernel)
		{
			OrderedSpikes = orderedSpikes;
			Spikes3D = spikeData;

			List<List<float[,]>> potentials = new List<List<float[,]>>();
			Console.Write($"Testing kernel {kernel.Name}...");

			for (int i = 0; i < OrderedSpikes.Count; i++)
			{
				potentials.Add(kernel.TestKernelForPotentials(OrderedSpikes[i], Spikes3D[i]));
			}

			Console.WriteLine(" Done.");
			return potentials;
		}

		//Extracts S2 maximum potentials for each image
		public List<List<float>> GetKernelPotentials(List<List<SpikeData>> orderedSpikes,
			List<List<SpikeData[,]>> spikeData, Kernel kernel)
		{
			List<List<float[,]>> potentials = TestNetworkForPotentials(orderedSpikes, spikeData, kernel);

			Console.Write("Pooling potentials...");
			List<List<float>> potentials_pooled = new List<List<float>>(potentials.Count);
			for (int img = 0; img < potentials.Count; img++)
			{
				List<float> p = Enumerable.Repeat(0f, kernel.NumberOfFeature).ToList();
				for (int ftr = 0; ftr < potentials[img].Count; ++ftr)
				{
					for (int row = 0; row < potentials[img][ftr].GetLength(0); ++row)
					{
						for (int col = 0; col < potentials[img][ftr].GetLength(1); ++col)
						{
							if (p[ftr] < potentials[img][ftr][row, col])
							{
								p[ftr] = potentials[img][ftr][row, col];
							}
						}
					}
				}
				potentials_pooled.Add(p);
			}
			Console.WriteLine(" Done.");
			return potentials_pooled;
		}

		public void SaveKernel(string taskName, Kernel kernel, List<float[,]> preRealFeatures)
		{
			//saving
			kernel.ComputeRealFeatures(preRealFeatures);
			Console.Write($"Saving kernel {kernel.Name}...");
			kernel.SaveKernel(Path.Combine(taskName,$"{kernel.Name}.kernel"));
			Console.WriteLine(" Done.");

			SaveKernelFeatures(taskName, kernel);
			//SaveKernelMergedWeights(taskName, kernel);
			//SaveKernelWeights(taskName, kernel);
		}

		//Save synaptic weights of S2 in a text file
		//Note that only the maximum weight among synapses corresponding to orientations will be saved
		public static void SaveKernelMergedWeights(string taskName, Kernel ker)
		{
			Console.Write("Saving merged weights for kernel " + ker.Name + " to file...");
			if(!Directory.Exists(Path.Combine(taskName, $"{ker.Name}_MergedWeights")))
			{
				Directory.CreateDirectory(Path.Combine(taskName, $"{ker.Name}_MergedWeights"));
			}

			for (int i = 0; i < ker.Weights.Count; i++)
			{
				string pw = Path.Combine($"{ker.Name}_MergedWeights", ker.Name + "_SumWeights_n" + i + ".txt");
				pw = Path.Combine(taskName, pw);
				StreamWriter sumWriter = new StreamWriter(pw);

				for (int j = 0; j < ker.Weights[i].GetLength(0); j++)
				{
					for (int k = 0; k < ker.Weights[i].GetLength(1); k++)
					{
						float sum = 0;
						for (int m = 0; m < ker.Weights[i].GetLength(2); m++)
						{
							sum += ker.Weights[i][j, k, m];
						}
						if (sum > 1)
							sum = 1;
						sumWriter.Write(sum + " ");
					}
					sumWriter.WriteLine();
				}

				sumWriter.Close();
			}
			Console.WriteLine(" Done.");

			Console.Write("Generating gnuplot file...");
			string p = Path.Combine($"{ker.Name}_MergedWeights", ker.Name + "_SumWeightsPlot.plt");
			p = Path.Combine(taskName, p);
			StreamWriter gnuWriter = new StreamWriter(p);
			gnuWriter.WriteLine($"set xrange [-0.5:{ker.Weights[0].GetLength(1) - 0.5}]; set yrange [{ker.Weights[0].GetLength(0) - 0.5}:-0.5]");
			//gnuWriter.WriteLine($"set xrange [-0.5:{ker.RealFeatures[0].GetLength(1) - 0.5}]; set yrange [{ker.RealFeatures[0].GetLength(0) - 0.5}:-0.5]");
			gnuWriter.WriteLine("set size ratio 1");
			gnuWriter.WriteLine("set cbrange [0:1]");
			gnuWriter.WriteLine("set pm3d map");
			gnuWriter.WriteLine("unset colorbox");
			gnuWriter.WriteLine("set palette gray");
			gnuWriter.WriteLine("set terminal png");

			gnuWriter.WriteLine("do for [i = 0:" + (ker.NumberOfFeature - 1) + "]{");
			gnuWriter.WriteLine($"\tt = sprintf('Sum Weights | Kernel: {ker.Name}, Neuron: %d', i)");
			//gnuWriter.WriteLine($"\tt = sprintf('Feature | Kernel: {ker.Name}, Neuron: %d', i)");
			gnuWriter.WriteLine("\tset title t");
			gnuWriter.WriteLine($"\toutfile = sprintf('{ker.Name}_SumWeights_n%03.0f.png', i)");
			gnuWriter.WriteLine("\tset output outfile");
			gnuWriter.WriteLine($"\tinfile = sprintf('{ker.Name}_SumWeights_n%d.txt', i)");
			gnuWriter.WriteLine("\tsplot infile matrix with image");
			gnuWriter.WriteLine("}");
			gnuWriter.Close();
			Console.WriteLine(" Done.");
		}

		//Save deconvolution of weights of S2 in a text file (for visualization)
		public static void SaveKernelFeatures(string taskName, Kernel ker)
		{
			Console.Write("Saving features for kernel " + ker.Name + " to file...");
			if (!Directory.Exists(Path.Combine(taskName, $"{ker.Name}_Features")))
			{
				Directory.CreateDirectory(Path.Combine(taskName, $"{ker.Name}_Features"));
			}

			for (int i = 0; i < ker.Weights.Count; i++)
			{
				string pw = Path.Combine($"{ker.Name}_Features", ker.Name + "_Features_n" + i + ".txt");
				pw = Path.Combine(taskName, pw);
				StreamWriter sumWriter = new StreamWriter(pw);

				for (int j = 0; j < ker.RealFeatures[i].GetLength(0); j++)
				{
					for (int k = 0; k < ker.RealFeatures[i].GetLength(1); k++)
					{
						sumWriter.Write(ker.RealFeatures[i][j, k] + " ");
					}
					sumWriter.WriteLine();
				}
				sumWriter.Close();
			}
			Console.WriteLine(" Done.");

			Console.Write("Generating gnuplot file...");
			string p = Path.Combine($"{ker.Name}_Features", ker.Name + "_FeaturesPlot.plt");
			p = Path.Combine(taskName, p);

			StreamWriter gnuWriter = new StreamWriter(p);

			gnuWriter.WriteLine($"set xrange [-0.5:{ker.RealFeatures[0].GetLength(1) - 0.5}]; set yrange [{ker.RealFeatures[0].GetLength(0) - 0.5}:-0.5]");
			gnuWriter.WriteLine("set size ratio 1");
			gnuWriter.WriteLine("set cbrange [0:1]");
			gnuWriter.WriteLine("set pm3d map");
			gnuWriter.WriteLine("unset colorbox");
			gnuWriter.WriteLine("set palette gray");
			gnuWriter.WriteLine("set terminal pngcairo enhanced crop");
			gnuWriter.WriteLine("unset xtics");
			gnuWriter.WriteLine("unset ytics");

			gnuWriter.WriteLine("do for [i = 0:" + (ker.NumberOfFeature - 1) + "]{");
			gnuWriter.WriteLine($"\toutfile = sprintf('{ker.Name}_Features_n%03.0f.png', i)");
			gnuWriter.WriteLine("\tset output outfile");
			gnuWriter.WriteLine($"\tinfile = sprintf('{ker.Name}_Features_n%d.txt', i)");
			gnuWriter.WriteLine("\tsplot infile matrix with image");
			gnuWriter.WriteLine("}");
			gnuWriter.Close();
			Console.WriteLine(" Done.");
		}

		//Save synaptic weights of S2 in a text file
		public static void SaveKernelWeights(string taskName, Kernel ker)
		{
			Console.WriteLine("Saving kernel weights to file...");
			if (!Directory.Exists(Path.Combine(taskName, $"{ker.Name}_Weights")))
			{
				Directory.CreateDirectory(Path.Combine(taskName, $"{ker.Name}_Weights"));
			}
			for (int i = 0; i < ker.Weights.Count; i++)
			{
				string pw = Path.Combine($"{ker.Name}_Weights", ker.Name + "_Weights_n" + i + ".txt");
				pw = Path.Combine(taskName, pw);
				StreamWriter writer = new StreamWriter(pw);
				for (int j = 0; j < ker.Weights[i].GetLength(0); j++)
				{
					for (int m = 0; m < ker.Weights[i].GetLength(2); m++)
					{
						for (int k = 0; k < ker.Weights[i].GetLength(1); k++)
						{
							writer.Write(ker.Weights[i][j, k, m] + " ");
						}
					}
					writer.WriteLine();
				}
				writer.Close();
			}

			Console.Write("Generating gnuplot file...");
			string p = Path.Combine($"{ker.Name}_Weights", ker.Name + "_WeightsPlot.plt");
			p = Path.Combine(taskName, p);
			StreamWriter gnuWriter = new StreamWriter(p);
			gnuWriter.WriteLine($"set xrange [-0.5:{ker.Weights[0].GetLength(2) * ker.Weights[0].GetLength(1) - 0.5}]; set yrange [{ker.Weights[0].GetLength(0) - 0.5}:-0.5]");
			gnuWriter.WriteLine($"set size ratio {1f / ker.Weights[0].GetLength(2)}");
			gnuWriter.WriteLine("set cbrange [0:1]");
			gnuWriter.WriteLine("set pm3d map");
			gnuWriter.WriteLine("unset colorbox");
			gnuWriter.WriteLine("set palette gray");
			gnuWriter.WriteLine("set terminal png");

			gnuWriter.WriteLine("do for [i = 0:" + (ker.NumberOfFeature - 1) + "]{");
			gnuWriter.WriteLine($"\tt = sprintf('Weights | Kernel: {ker.Name}, Neuron: %d', i)");
			gnuWriter.WriteLine("\tset title t");
			gnuWriter.WriteLine($"\toutfile = sprintf('{ker.Name}_Weights_n%03.0f.png', i)");
			gnuWriter.WriteLine("\tset output outfile");
			gnuWriter.WriteLine($"\tinfile = sprintf('{ker.Name}_Weights_n%d.txt', i)");
			gnuWriter.WriteLine("\tsplot infile matrix with image");
			gnuWriter.WriteLine("}");
			gnuWriter.Close();
			Console.WriteLine(" Done.");
			Console.WriteLine(" Done.");
		}
	}
}
