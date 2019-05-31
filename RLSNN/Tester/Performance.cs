using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TimTester
{
	class Performance
	{
		public float Corrects { get; private set; }
		public float Wrongs { get; private set; }
		public float Silences { get; private set; }
		public int Iteration { get; private set; }

		public Performance(int iteration, int corrects, int wrongs, int silences)
		{
			Iteration = iteration;
			int total = corrects + wrongs + silences;
			Corrects = ((float)corrects) / total;
			Wrongs = ((float)wrongs) / total;
			Silences = ((float)silences) / total;
		}

		public override string ToString()
		{
			return $"Iteration: {Iteration}, Correct: {Corrects}, Wrong: {Wrongs}, Silence: {Silences}";
		}
	}
}
