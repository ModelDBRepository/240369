using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core
{
	[Serializable]
	//Generates and applies Gabor filters
	public class Gabor
	{
		public float Scale { get; set; }
		public float[] Orientations { get; set; }
		public int RFSize { get; set; }
		public float Div { get; set; }

		public List<float[,]> gaborFilters;
		public List<float[,]> gaborFeatures;

		public Gabor(float scale, float[] orientations, int rfSize, float div)
		{
			Scale = scale;
			Orientations = orientations;
			RFSize = rfSize;
			Div = div;

			GenerateGaborFilters();
			GenerateGaborFeatures();
		}

		//Generates a cleaner view for Gabor filters for the sake of visualization
		private void GenerateGaborFeatures()
		{
			gaborFeatures = new List<float[,]>(gaborFilters.Count);
			for(int i = 0; i < gaborFilters.Count; i++)
			{
				float maxi = gaborFilters[i].Cast<float>().Max();
				gaborFeatures.Add(new float[gaborFilters[i].GetLength(0), gaborFilters[i].GetLength(1)]);
				for(int r = 0; r < gaborFeatures[i].GetLength(0); r++)
				{
					for(int c = 0; c < gaborFeatures[i].GetLength(1); c++)
					{
						if(Math.Abs(gaborFilters[i][r,c] - maxi) <= 0.1)
						{
							gaborFeatures[i][r, c] = gaborFilters[i][r, c];
						}
						else
						{
							gaborFeatures[i][r, c] = 0;
						}
					}
				}
			}
		}

		//Generates Gabor filters for all of the orientations
		private void GenerateGaborFilters()
		{
			gaborFilters = new List<float[,]>(Orientations.Length);
			float lambda = RFSize * 2 / Div;
			float sigma = lambda * 0.8f;
			float sigmaSq = sigma * sigma;
			float g = 0.3f;
			int offset = RFSize / 2;
			for (int ori = 0; ori < Orientations.Length; ori++)
			{
				gaborFilters.Add(new float[RFSize, RFSize]);
				float sumSq = 0;
				float sum = 0;

				float theta = (Orientations[ori] * (float)Math.PI) / 180;
				for (int i = -offset; i <= offset; i++)
				{
					for (int j = -offset; j <= offset; j++)
					{
						float value = 0;
						if (Math.Sqrt(i * i + j * j) <= RFSize / 2f)
						{
							float x = (float)(i * Math.Cos(theta) - j * Math.Sin(theta));
							float y = (float)(i * Math.Sin(theta) + j * Math.Cos(theta));
							value = (float)(Math.Exp(-(x * x + g * g * y * y) / (2 * sigmaSq))) *
								(float)Math.Cos(2 * Math.PI * x / lambda);
							sum += value;
							sumSq += value * value;
						}
						gaborFilters[ori][i + offset, offset - j] = value;
					}
				}

				float mean = sum / (RFSize * RFSize);
				sumSq = (float)Math.Sqrt(sumSq);
				for (int i = -offset; i <= offset; i++)
				{
					for (int j = -offset; j <= offset; j++)
					{
						gaborFilters[ori][i + offset, j + offset] -= mean;
						gaborFilters[ori][i + offset, j + offset] /= sumSq;
					}
				}
			}
		}

		//Computes spike times according to Gabor values for an image
		public List<SpikeData[,]> GetGaboredTimes(string imageAddress)
		{
			List<SpikeData[,]> spike2D = new List<SpikeData[,]>(Orientations.Length);
			float[,] scaledImage;
			float[,] normedImage;
			List<float[,]> gaboredImages = new List<float[,]>(Orientations.Length);
			int offset = RFSize / 2;

			//computing normal images
			scaledImage = Utility.ReadImageGrayScale(imageAddress, Scale);
			normedImage = new float[scaledImage.GetLength(0), scaledImage.GetLength(1)];

			//computing normals
			for (int i = 0; i < scaledImage.GetLength(0); i++)
			{
				for (int j = 0; j < scaledImage.GetLength(1); j++)
				{
					for (int row = -offset; row <= offset; row++)
					{
						if (i + row >= 0 && i + row < scaledImage.GetLength(0))
						{
							for (int col = -offset; col <= offset; col++)
							{
								if (j + col >= 0 && j + col < scaledImage.GetLength(1))
								{
									normedImage[i, j] +=
										scaledImage[i + row, j + col] * scaledImage[i + row, j + col];
								}
							}
						}
					}
					normedImage[i, j] = (float)Math.Sqrt(normedImage[i, j]);
					if (normedImage[i, j] == 0)
						normedImage[i, j] = 1;
				}
			}

			//computing Gabor values for each orientation and its spike latencies
			for (int ori = 0; ori < Orientations.Length; ori++)
			{
				gaboredImages.Add(new float[scaledImage.GetLength(0), scaledImage.GetLength(1)]);
				spike2D.Add(new SpikeData[scaledImage.GetLength(0), scaledImage.GetLength(1)]);
				for (int i = offset; i < scaledImage.GetLength(0) - offset; i++)
				{
					for (int j = offset; j < scaledImage.GetLength(1) - offset; j++)
					{
						for (int row = i - offset, fr = 0; row <= i + offset; row++, fr++)
						{
							for (int col = j - offset, fc = 0; col <= j + offset; col++, fc++)
							{
								gaboredImages[ori][i, j] +=
									gaborFilters[ori][fr, fc] * scaledImage[row, col];
							}
						}
						gaboredImages[ori][i, j] = Math.Abs(gaboredImages[ori][i, j]);
						gaboredImages[ori][i, j] /= normedImage[i, j];
						if (gaboredImages[ori][i, j] > 0)
						{
							spike2D[ori][i, j] = new SpikeData(1 / gaboredImages[ori][i, j],
								i, j, ori);
						}
					}
				}
			}
			return spike2D;
		}
	}
}
