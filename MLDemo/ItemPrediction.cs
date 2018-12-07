using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace MLDemo
{
    public class ItemPrediction
    {
        [ColumnName("PredictedCatID")]
        public string CategoryID;
    }
}
