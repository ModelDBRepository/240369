using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Core
{
	[Serializable]
	//Holds all of the Gabor-applied input images
	public class GaborLayer
	{
		public int ImageIdx { get; set; }
		public string[] FileNames { get; set; }
		public int NumberOfImages { get { return FileNames.Length; } }
		public float[] InhibitionPercents { get; set; }
		public List<List<SpikeData>> OrderedData { get; set; }	//Ordered spikes for each image to be porpagated
		public List<List<SpikeData[,]>> SpikeData2D { get; set; }	//Spike information for each orientation of all images
		public Gabor gabor { get; set; }

		public GaborLayer(string[] folderAddresses, float scale, float[] orientations, int rfSize, float div,
			int cRF, float[] inhibitionPercents)
		{
			gabor = new Gabor(scale, orientations, rfSize, div);
			InhibitionPercents = inhibitionPercents;
			
			List<string> allFileNames = new List<string>();
			foreach (string folderAddress in folderAddresses)
			{
				allFileNames.AddRange(Directory.GetFiles(folderAddress, "*.png"));
			}
			FileNames = allFileNames.ToArray();

			OrderedData = new List<List<SpikeData>>(NumberOfImages);
			SpikeData2D = new List<List<SpikeData[,]>>(NumberOfImages);
			ImageIdx = 0;
			while (ImageIdx < NumberOfImages)
			{
				Console.Write($"{ImageIdx}: {FileNames[ImageIdx]}... ");
				List<SpikeData[,]> temp;
				OrderedData.Add(GetGaboredTimes(FileNames[ImageIdx], out temp, cRF));
				SpikeData2D.Add(temp);
				ImageIdx++;
				Console.WriteLine("Done.");
			}
		}

		public GaborLayer(List<string> imageAddresses, float scale, float[] orientations, int rfSize, float div,
			int cRF, float[] inhibitionPercents)
		{
			gabor = new Gabor(scale, orientations, rfSize, div);
			
			InhibitionPercents = inhibitionPercents;
			FileNames = imageAddresses.ToArray();

			OrderedData = new List<List<SpikeData>>(NumberOfImages);
			SpikeData2D = new List<List<SpikeData[,]>>(NumberOfImages);
			ImageIdx = 0;
			while (ImageIdx < NumberOfImages)
			{
				Console.Write($"{ImageIdx}: {FileNames[ImageIdx]}... ");
				List<SpikeData[,]> temp;
				OrderedData.Add(GetGaboredTimes(FileNames[ImageIdx], out temp, cRF));
				SpikeData2D.Add(temp);
				ImageIdx++;
				Console.WriteLine("Done.");
			}
		}
		
		//Applies competition between orientations (for each position, only the most fitted orientation can
		//emit a spike)
		private List<SpikeData> GetGaboredTimes(string imageAddress, out List<SpikeData[,]> spike2DOriPooled,
			int complexField)
		{
			List<SpikeData[,]> spike2D = gabor.GetGaboredTimes(imageAddress);
			if (complexField > 1)
				spike2DOriPooled = GetPooledTimes(complexField, spike2D);
			else
				spike2DOriPooled = spike2D;
			ApplyLateralInhibition(spike2DOriPooled);
			
			List<SpikeData> result = new List<SpikeData>();
			//Finding best fitted orientation
			for (int r = 0; r < spike2DOriPooled[0].GetLength(0); r++)
			{
				for (int c = 0; c < spike2DOriPooled[0].GetLength(1); c++)
				{
					int miniIdx = 0;
					SpikeData mini = spike2DOriPooled[0][r, c];
					for (int f = 1; f < spike2DOriPooled.Count; f++)
					{
						if (spike2DOriPooled[f][r, c] != null)
						{
							if (mini == null || spike2DOriPooled[f][r, c].Time < mini.Time)
							{
								mini = spike2DOriPooled[f][r, c];
								miniIdx = f;
							}
						}
					}
					if (mini != null)
					{
						result.Add(mini);
						for (int f = 0; f < miniIdx; f++)
						{
							spike2DOriPooled[f][r, c] = null;
						}
						for (int f = miniIdx + 1; f < spike2DOriPooled.Count; f++)
						{
							spike2DOriPooled[f][r, c] = null;
						}
					}
				}
			}
			
			//Sorting spikes with respect to their latencies
			result = result.OrderBy(sd => sd.Time).ToList();
			return result;
		}
		
		//Applies local lateral inhibition on each orientation
		private void ApplyLateralInhibition(List<SpikeData[,]> spikes3D)
		{
			int inhibOffset = InhibitionPercents.Length;

			//computing lateral inhibition
			for (int f = 0; f < spikes3D.Count; f++)
			{
				for (int row = 0; row < spikes3D[f].GetLength(0); row++)
				{
					int minRow = Math.Max(0, row - inhibOffset);
					int maxRow = Math.Min(row + inhibOffset + 1, spikes3D[f].GetLength(0));

					for (int col = 0; col < spikes3D[f].GetLength(1); col++)
					{
						if (spikes3D[f][row, col] != null)
						{
							spikes3D[f][row, col].InhibitedTime = spikes3D[f][row, col].Time;
							int minCol = Math.Max(0, col - inhibOffset);
							int maxCol = Math.Min(col + inhibOffset + 1, spikes3D[f].GetLength(1));

							for (int r = minRow; r < maxRow; r++)
							{
								for (int c = minCol; c < maxCol; c++)
								{
									if (spikes3D[f][r, c] != null)
									{
										int dist = Math.Max(Math.Abs(row - r), Math.Abs(col - c));
										if (dist > 0 && spikes3D[f][r, c].Time <
											spikes3D[f][row, col].Time)
										{
											spikes3D[f][row, col].InhibitedTime +=
												spikes3D[f][row, col].Time * InhibitionPercents[dist - 1];
										}
									}
								}
							}
						}
					}
				}
			}

			//applying computed inhibitions
			for (int f = 0; f < spikes3D.Count; f++)
			{
				for (int row = 0; row < spikes3D[f].GetLength(0); row++)
				{
					for (int col = 0; col < spikes3D[f].GetLength(1); col++)
					{
						if (spikes3D[f][row, col] != null)
						{
							spikes3D[f][row, col].Time = spikes3D[f][row, col].InhibitedTime;
						}
					}
				}
			}
		}

		//Applies pooling on each orientation
		private List<SpikeData[,]> GetPooledTimes(int complexField, List<SpikeData[,]> spikes3D)
		{
			int CStride = complexField - 1;
			int CRF = complexField;
			List<SpikeData[,]> spikes3DPooled = new List<SpikeData[,]>(spikes3D.Count);

			//pooling on each feature

			for (int f = 0; f < spikes3D.Count; f++)
			{
				spikes3DPooled.Add(new SpikeData[
					(int)Math.Ceiling((double)spikes3D[f].GetLength(0) / CStride),
					(int)Math.Ceiling((double)spikes3D[f].GetLength(1) / CStride)]);

				for (int r = 0; r < spikes3DPooled[f].GetLength(0); r++)
				{
					int minRow = r * CStride;
					int maxRow = Math.Min(minRow + CRF, spikes3D[f].GetLength(0));

					for (int c = 0; c < spikes3DPooled[f].GetLength(1); c++)
					{
						int minCol = c * CStride;
						int maxCol = Math.Min(minCol + CRF, spikes3D[f].GetLength(1));

						SpikeData mini = null;
						for (int rr = minRow; rr < maxRow; rr++)
						{
							for (int cc = minCol; cc < maxCol; cc++)
							{
								if (spikes3D[f][rr, cc] != null)
								{
									if (mini == null || spikes3D[f][rr, cc].Time < mini.Time)
									{
										mini = spikes3D[f][rr, cc];
									}
								}
							}
						}
						if (mini != null)
						{
							spikes3DPooled[f][r, c] = new SpikeData(mini.Time, r, c, mini.Feature);
						}
					}
				}
			}

			return spikes3DPooled;
		}

		//Shuffles order of input images
		private void ShuffleImages()
		{
			Random rnd = new Random();
			int n = OrderedData.Count;
			while (n > 1)
			{
				n--;
				int k = rnd.Next(n + 1);
				var value1 = OrderedData[k];
				var value2 = SpikeData2D[k];
				OrderedData[k] = OrderedData[n];
				SpikeData2D[k] = SpikeData2D[n];
				OrderedData[n] = value1;
				SpikeData2D[n] = value2;
			}
		}

		public void SaveGaborLayer(string address)
		{
			IFormatter formatter = new BinaryFormatter();
			Stream stream = new FileStream(address,
									 FileMode.Create,
									 FileAccess.Write, FileShare.None);
			formatter.Serialize(stream, this);
			stream.Close();
		}

		public static GaborLayer LoadGaborLayer(string address)
		{
			IFormatter formatter = new BinaryFormatter();
			Stream stream = new FileStream(address,
				FileMode.Open,
				FileAccess.Read,
				FileShare.Read);
			GaborLayer result = (GaborLayer)formatter.Deserialize(stream);
			stream.Close();
			return result;
		}
	}
}