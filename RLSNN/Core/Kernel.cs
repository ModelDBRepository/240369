using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

namespace Core
{
	[Serializable]
	//Manages S2 layer
	public class Kernel
	{
		public string Name { get; set; }		//Needed when saving on secondary memory
		public int NumberOfFeature { get; set; }	//Number of features to be extracted
		public List<float[,,]> Weights { get; set; }	//Weight matrix for each feature
		public List<int[,,]> DeltaWeights { get; set; }		//Weight changes for each feature
		public List<float[,]> RealFeatures { get; set; }	//Appearence of a feature after deconvolution
		public int SRF { get; set; }	//Input window size
		public float[] Thresholds { get; set; }		//Thresholds of neurons for each feature
		public int KWTA { get; set; }		//Number of STDP winners for each image presentation

		public float Ap { get; set; }		//Equivalent to A_r^+ in the paper
		public float An { get; set; }       //Equivalent to A_r^- in the paper
		public float AntiAp { get; set; }   //Equivalent to A_p^- in the paper
		public float AntiAn { get; set; }	//Equivalent to A_p^+ in the paper

		private int sOffset;

		private int sOffsetDiv2;
		private Random random;
		private static Random tieBreaker = new Random(40);

		private static float weightOffset = 0.00001f;

		public Kernel(string name, int numberOfFeatures, int numberOfOrientations,
			int srf, float[] thresholds,
			int kwta, float ap, float an, float antiAp, float antiAn,
			float meanWeight, float stdDevWeight, Random rand)
		{
			InitializeKernelWithoutWeights(name, numberOfFeatures,
				srf, thresholds, kwta, ap, an, antiAp, antiAn);

			random = rand;
			Weights = new List<float[,,]>(NumberOfFeature);
			DeltaWeights = new List<int[,,]>(NumberOfFeature);
			for (int i = 0; i < NumberOfFeature; i++)
			{
				Weights.Add(new float[SRF, SRF, numberOfOrientations]);
				DeltaWeights.Add(new int[SRF, SRF, numberOfOrientations]);
				for (int r = 0; r < SRF; r++)
				{
					for (int c = 0; c < SRF; c++)
					{
						for (int p = 0; p < numberOfOrientations; p++)
						{
							double u1 = rand.NextDouble(); //these are uniform(0,1) random doubles
							double u2 = rand.NextDouble();
							double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
										 Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
							double randNormal =
										 meanWeight + stdDevWeight * randStdNormal; //random normal(mean,stdDev^2)

							Weights[i][r, c, p] = (float)randNormal;
							DeltaWeights[i][r, c, p] = 0;
						}
					}
				}
			}
		}

		public Kernel(string name, int numberOfFeatures, List<float[,,]> weights, List<int[,,]> deltaWeights,
			int srf, float[] thresholds,
			int kwta, float ap, float an, float antiAp, float antiAn)
		{
			InitializeKernelWithoutWeights(name, numberOfFeatures,
				srf, thresholds, kwta, ap, an, antiAp, antiAn);

			Weights = weights;
			DeltaWeights = deltaWeights;
		}

		private void InitializeKernelWithoutWeights(string name, int numberOfFeatures,
			int srf, float[] thresholds,
			int kwta, float ap, float an, float antiAp, float antiAn)
		{
			Name = name;
			NumberOfFeature = numberOfFeatures;
			SRF = srf;
			sOffset = SRF / 2;
			sOffsetDiv2 = sOffset;
			Thresholds = thresholds;
			KWTA = kwta;

			Ap = ap;
			An = an;
			AntiAp = antiAp;
			AntiAn = antiAn;
		}
		
		//Retrieves earliest spikes of Layer S2 for an input stimuli
		public List<SpikeData> TestKernelForFirstSpikes(List<SpikeData> spikes, List<SpikeData[,]> spikes2DIn)
		{
			List<SpikeData> winners = new List<SpikeData>();

			List<float[,]> potentials = new List<float[,]>(spikes2DIn.Count);
			bool[] hasFired = new bool[NumberOfFeature];
			int fireCount = 0;

			//initializing
			int rows = spikes2DIn[0].GetLength(0);
			int cols = spikes2DIn[0].GetLength(1);

			for (int j = 0; j < NumberOfFeature; j++)
			{
				potentials.Add(new float[rows, cols]);
			}
			
			int rowLowerBound = sOffsetDiv2;
			int colLowerBound = sOffsetDiv2;
			int rowUpperBound = potentials[0].GetLength(0) - sOffsetDiv2;
			int colUpperBound = potentials[0].GetLength(1) - sOffsetDiv2;

			for (int i = 0; i < spikes.Count; i++)
			{
				SpikeData current = spikes[i];
				if (fireCount < KWTA)
				{
					//Finding affected neurons range
					int minRow = Math.Max(rowLowerBound, current.Row - sOffset);
					int maxRow = Math.Min(rowUpperBound, current.Row + sOffset + 1);

					int minCol = Math.Max(colLowerBound, current.Column - sOffset);
					int maxCol = Math.Min(colUpperBound, current.Column + sOffset + 1);

					for (int j = 0; j < NumberOfFeature; j++)
					{
						//updating
						for (int r = minRow; fireCount < KWTA && !hasFired[j] && r < maxRow; r++)
						{
							for (int c = minCol; fireCount < KWTA && !hasFired[j] && c < maxCol; c++)
							{
								potentials[j][r, c] += getWeight(j, (current.Row - r),
									(current.Column - c), current.Feature);

								//check fire
								if (potentials[j][r, c] >= Thresholds[j])
								{
									++fireCount;
									hasFired[j] = true;
									winners.Add(new SpikeData(current.Time, r, c, j));
								}
							}
						}
					}
				}
			}
			return winners;
		}

		//Retrieves potentials of Layer S2 for an input stimuli
		public List<float[,]> TestKernelForPotentials(List<SpikeData> spikes, List<SpikeData[,]> spikes2DIn)
		{
			List<float[,]> potentials = new List<float[,]>(spikes2DIn.Count);

			//initializing
			int rows = spikes2DIn[0].GetLength(0);
			int cols = spikes2DIn[0].GetLength(1);

			for (int j = 0; j < NumberOfFeature; j++)
			{
				potentials.Add(new float[rows, cols]);
			}
			
			int rowLowerBound = sOffsetDiv2;
			int colLowerBound = sOffsetDiv2;
			int rowUpperBound = potentials[0].GetLength(0) - sOffsetDiv2;
			int colUpperBound = potentials[0].GetLength(1) - sOffsetDiv2;

			//S firings
			for (int i = 0; i < spikes.Count; i++)
			{
				SpikeData current = spikes[i];
				int minRow = Math.Max(rowLowerBound, current.Row - sOffset);
				int maxRow = Math.Min(rowUpperBound, current.Row + sOffset + 1);

				int minCol = Math.Max(colLowerBound, current.Column - sOffset);
				int maxCol = Math.Min(colUpperBound, current.Column + sOffset + 1);

				for (int j = 0; j < NumberOfFeature; j++)
				{
					//updating
					for (int r = minRow; r < maxRow; r++)
					{
						for (int c = minCol; c < maxCol; c++)
						{
							potentials[j][r, c] += getWeight(j, (current.Row - r),
								(current.Column - c), current.Feature);
						}
					}
				}
			}

			return potentials;
		}

		//Trains Layer S2 with an input stimuli and returns the STDP winners
		//(Only marks for STDP. Applying weight changes will happen later)
		public List<SpikeData> TrainKernelAndGetSTDPWinners(List<SpikeData> spikes, List<SpikeData[,]> spikes2D,
			bool[] isActive = null)
		{
			List<SpikeData> winners = new List<SpikeData>();

			List<float[,]> potentials = new List<float[,]>(spikes2D.Count);
			bool[] hasFired = new bool[NumberOfFeature];
			int fireCount = 0;

			//initializing
			int rows = spikes2D[0].GetLength(0);
			int cols = spikes2D[0].GetLength(1);

			for (int j = 0; j < NumberOfFeature; j++)
			{
				potentials.Add(new float[rows, cols]);
			}

			int rowLowerBound = sOffsetDiv2;
			int colLowerBound = sOffsetDiv2;
			int rowUpperBound = potentials[0].GetLength(0) - sOffsetDiv2;
			int colUpperBound = potentials[0].GetLength(1) - sOffsetDiv2;

			for (int i = 0; i < spikes.Count; i++)
			{
				SpikeData current = spikes[i];
				if (fireCount < KWTA)
				{
					//Finding affected neurons range
					int minRow = Math.Max(rowLowerBound, current.Row - sOffset);
					int maxRow = Math.Min(rowUpperBound, current.Row + sOffset + 1);

					int minCol = Math.Max(colLowerBound, current.Column - sOffset);
					int maxCol = Math.Min(colUpperBound, current.Column + sOffset + 1);

					for (int j = 0; j < NumberOfFeature; j++)
					{
						if (isActive != null && !isActive[j]) continue;
						//updating
						for (int r = minRow; fireCount < KWTA && !hasFired[j] && r < maxRow; r++)
						{
							for (int c = minCol; fireCount < KWTA && !hasFired[j] && c < maxCol; c++)
							{
								potentials[j][r, c] += getWeight(j, (current.Row - r),
									(current.Column - c), current.Feature);

								//check fire
								if (potentials[j][r, c] >= Thresholds[j])
								{
									++fireCount;
									hasFired[j] = true;
									winners.Add(new SpikeData(current.Time, r, c, j));
									//Set STDP
									SetSTDP(j, r, c, spikes2D, current);
								}
							}
						}
					}
				}
			}
			return winners;
		}

		//Trains Layer S2 with an input stimuli
		//(Only marks for STDP. Applying weight changes will happen later)
		public void TrainKernel(List<SpikeData> spikes, List<SpikeData[,]> spikes2D)
		{
			List<float[,]> potentials = new List<float[,]>(spikes2D.Count);
			bool[] hasFired = new bool[NumberOfFeature];
			int fireCount = 0;

			//initializing
			int rows = spikes2D[0].GetLength(0);
			int cols = spikes2D[0].GetLength(1);

			for (int j = 0; j < NumberOfFeature; j++)
			{
				potentials.Add(new float[rows, cols]);
			}

			int rowLowerBound = sOffsetDiv2;
			int colLowerBound = sOffsetDiv2;
			int rowUpperBound = potentials[0].GetLength(0) - sOffsetDiv2;
			int colUpperBound = potentials[0].GetLength(1) - sOffsetDiv2;

			for (int i = 0; i < spikes.Count; i++)
			{
				SpikeData current = spikes[i];
				if (fireCount < KWTA)
				{
					//Finding affected neurons range
					int minRow = Math.Max(rowLowerBound, current.Row - sOffset);
					int maxRow = Math.Min(rowUpperBound, current.Row + sOffset + 1);

					int minCol = Math.Max(colLowerBound, current.Column - sOffset);
					int maxCol = Math.Min(colUpperBound, current.Column + sOffset + 1);

					for (int j = 0; j < NumberOfFeature; j++)
					{
						//updating
						for (int r = minRow; fireCount < KWTA && !hasFired[j] && r < maxRow; r++)
						{
							for (int c = minCol; fireCount < KWTA && !hasFired[j] && c < maxCol; c++)
							{
								potentials[j][r, c] += getWeight(j, (current.Row - r),
									(current.Column - c), current.Feature);

								//check fire
								if (potentials[j][r, c] >= Thresholds[j])
								{
									++fireCount;
									hasFired[j] = true;
									//Set STDP
									SetSTDP(j, r, c, spikes2D, current);
								}
							}
						}
					}
				}
			}
		}

		//Updates weight change matrix
		private void SetSTDP(int f, int r, int c, List<SpikeData[,]> spikes2D, SpikeData currentSpike)
		{
			for (int feature = 0; feature < Weights[f].GetLength(2); feature++)
			{
				for (int i = -sOffset; i <= sOffset; i++)
				{
					int row = r + i;
					for (int j = -sOffset; j <= sOffset; j++)
					{
						int col = c + j;
						if (row >= 0 && row < spikes2D[feature].GetLength(0) &&
							col >= 0 && col < spikes2D[feature].GetLength(1))
						{
							if (spikes2D[feature][row, col]?.Time <= currentSpike.Time)
							{
								SetIncreament(f, i, j, feature);
							}
							else
							{
								SetDecreament(f, i, j, feature);
							}
						}
						else
						{
							SetDecreament(f, i, j, feature);
						}
					}
				}
			}
		}
		
		//Applies weight changes according to the weight change matrix
		public void ApplySTDP(bool reward)
		{
			for (int feature = 0; feature < Weights.Count; ++feature)
			{
				for(int r = 0; r < Weights[feature].GetLength(0); ++r)
				{
					for(int c = 0; c < Weights[feature].GetLength(1); ++c)
					{
						for (int f = 0; f < Weights[feature].GetLength(2); ++f)
						{
							if (reward)
							{
								if (DeltaWeights[feature][r, c, f] > 0)
									Weights[feature][r, c, f] += (Ap * Weights[feature][r, c, f] * (1 - Weights[feature][r, c, f])) *
										DeltaWeights[feature][r, c, f];
								else if(DeltaWeights[feature][r, c, f] < 0)
									Weights[feature][r, c, f] += (An * Weights[feature][r, c, f] * (1 - Weights[feature][r, c, f])) *
										DeltaWeights[feature][r, c, f];
							}
							else
							{
								if (DeltaWeights[feature][r, c, f] > 0)
									Weights[feature][r, c, f] -= (AntiAp * Weights[feature][r, c, f] * (1 - Weights[feature][r, c, f])) *
										DeltaWeights[feature][r, c, f];
								else if (DeltaWeights[feature][r, c, f] < 0)
									Weights[feature][r, c, f] -= (AntiAn * Weights[feature][r, c, f] * (1 - Weights[feature][r, c, f])) *
										DeltaWeights[feature][r, c, f];
							}
							DeltaWeights[feature][r, c, f] = 0;

							if (Weights[feature][r, c, f] >= 1)
								Weights[feature][r, c, f] = 1 - weightOffset;
							else if (Weights[feature][r, c, f] <= 0)
								Weights[feature][r, c, f] = 0 + weightOffset;
						}
					}
				}
			}
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private float getWeight(int f, int r, int c, int pref)
		{
			return Weights[f][sOffset + r, sOffset + c, pref];
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void setWeight(int f, int r, int c, int pref, float value)
		{
			Weights[f][sOffset + r, sOffset + c, pref] = value;
		}
		
		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void SetIncreament(int f, int r, int c, int pref)
		{
			DeltaWeights[f][sOffset + r, sOffset + c, pref] += 1;
		}

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private void SetDecreament(int f, int r, int c, int pref)
		{
			DeltaWeights[f][sOffset + r, sOffset + c, pref] -= 1;
		}

		public void SaveKernel(string address)
		{
			IFormatter formatter = new BinaryFormatter();
			Stream stream = new FileStream(address,
									 FileMode.Create,
									 FileAccess.Write, FileShare.None);
			formatter.Serialize(stream, this);
			stream.Close();
		}

		public static Kernel LoadKernel(string address)
		{
			IFormatter formatter = new BinaryFormatter();
			Stream stream = new FileStream(address,
				FileMode.Open,
				FileAccess.Read,
				FileShare.Read);
			Kernel result =  (Kernel)formatter.Deserialize(stream);
			stream.Close();
			return result;
		}
		
		//Does the deconvolution of weight matrices to produce feature visualization
		public void ComputeRealFeatures(List<float[,]> preFeatures)
		{
			RealFeatures = new List<float[,]>(NumberOfFeature);
			int height = Weights[0].GetLength(0) * preFeatures[0].GetLength(0);
			int width = Weights[0].GetLength(1) * preFeatures[0].GetLength(1);

			List<float[,]> temp = new List<float[,]>(preFeatures.Count);
			for (int i = 0; i < preFeatures.Count; i++)
			{
				temp.Add(new float[height, width]);
			}

			int rowLength = preFeatures[0].GetLength(0);
			int colLength = preFeatures[0].GetLength(1);
			for (int f = 0; f < NumberOfFeature; f++)
			{
				RealFeatures.Add(new float[height, width]);
				for (int r = 0; r < Weights[f].GetLength(0); r++)
				{
					for (int c = 0; c < Weights[f].GetLength(1); c++)
					{
						float maxW = 0;
						int maxPF = 0;
						for (int pf = 0; pf < Weights[f].GetLength(2); pf++)
						{
							if(maxW < Weights[f][r, c, pf])
							{
								maxW = Weights[f][r, c, pf];
								maxPF = pf;
							}
						}

						for (int pr = 0; pr < rowLength; pr++)
						{
							for (int pc = 0; pc < colLength; pc++)
							{
								RealFeatures[f][rowLength * r + pr, colLength * c + pc] =
									Weights[f][r, c, maxPF] * preFeatures[maxPF][pr, pc];
							}
						}
					}
				}
			}
		}

		//Gets a deep copy of Layer S2
		public static Kernel GetCopy(Kernel kernel, string newName)
		{
			int numberOfPreKernelFeatures = kernel.Weights[0].GetLength(2);
			var weightsCopy = new List<float[,,]>(kernel.NumberOfFeature);
			var deltaWeightsCopy = new List<int[,,]>(kernel.NumberOfFeature);
			for (int i = 0; i < kernel.NumberOfFeature; i++)
			{
				weightsCopy.Add(new float[kernel.SRF, kernel.SRF, numberOfPreKernelFeatures]);
				deltaWeightsCopy.Add(new int[kernel.SRF, kernel.SRF, numberOfPreKernelFeatures]);
				for (int r = 0; r < kernel.SRF; r++)
				{
					for (int c = 0; c < kernel.SRF; c++)
					{
						for (int p = 0; p < numberOfPreKernelFeatures; p++)
						{
							weightsCopy[i][r, c, p] = kernel.Weights[i][r, c, p];
							deltaWeightsCopy[i][r, c, p] = kernel.DeltaWeights[i][r, c, p];
						}
					}
				}
			}

			List<float[,]> realFeaturesCopy = null;
			if (kernel.RealFeatures != null)
			{
				realFeaturesCopy = new List<float[,]>(kernel.RealFeatures.Count);
				foreach (float[,] rf in kernel.RealFeatures)
				{
					realFeaturesCopy.Add((float[,])rf.Clone());
				}
			}

			Kernel newKernel = new Kernel(newName,
				kernel.NumberOfFeature, weightsCopy, deltaWeightsCopy,
				kernel.SRF,
				(float[])kernel.Thresholds.Clone(), kernel.KWTA, kernel.Ap, kernel.An, kernel.AntiAp, kernel.AntiAn);
			newKernel.RealFeatures = realFeaturesCopy;

			return newKernel;
		}
	}
}
