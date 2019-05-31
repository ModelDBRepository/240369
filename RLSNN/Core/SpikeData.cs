using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Core
{
	[Serializable]
	//Holds information about spikes
	public class SpikeData
	{
		public float Time { get; set; }
		public int Row { get; set; }
		public int Column { get; set; }
		public int Feature { get; set; }
		public float InhibitedTime { get; set; } //just to hold temporary time

		public SpikeData(float time, int row, int col, int feature)
		{
			Time = time;
			Row = row;
			Column = col;
			Feature = feature;
		}
	}
}
