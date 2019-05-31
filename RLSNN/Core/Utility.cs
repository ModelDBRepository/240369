using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.IO;

namespace Core
{
	[Serializable]
	//Provides useful tools
	public class Utility
	{
		//Reads an image from memory and converts it to grayscale values
		public static float[,] ReadImageGrayScale(string imageAddress, float scale)
		{
			Bitmap image = new Bitmap(imageAddress);
			int scaledWidth = (int)(image.Width * scale);
			int scaledHeight = (int)(image.Height * scale);

			if (scale != 1)
			{
				image = new Bitmap(image, new Size(scaledWidth, scaledHeight));
			}
			
			float[,] gray = new float[image.Height, image.Width];
			for (int i = 0; i < image.Height; i++)
			{
				for (int j = 0; j < image.Width; j++)
				{
					Color c = image.GetPixel(j, i);
					gray[i, j] = ((c.R + c.B + c.G) / 3f) / 255f;
				}
			}
			return gray;
		}
	}
}
