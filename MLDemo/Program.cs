using System;
using System.IO;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System.Diagnostics;

namespace MLDemo
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string BaseDatasetsLocation = @"../../../../Data";
        private static string TrainDataPath = $"./Data/trainingdata.tsv";
        private static string TestDataPath = $"./Data/testingdata.tsv";
        private static string BaseModelsPath = @"../../../MLModels";
        private static string ModelPath = $"./Data/ItemCategorizationModel.zip";

        public enum MyTrainerStrategy : int
        {
            SdcaMultiClassTrainer = 1,
            OVAAveragedPerceptronTrainer = 2
        };

        static void Main(string[] args)
        {
            var ctx = new MLContext();

            var reader = ctx.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = "tab",
                HasHeader = false,
                Column = new[]
                {
                    new TextLoader.Column("CategoryID", DataKind.Text, 0),
                    new TextLoader.Column("Title", DataKind.Text, 1),
                    new TextLoader.Column("Description", DataKind.Text, 2),
                    }
            });

            //var traindata = reader.Read("/users/ryansmith/Projects/mldemo/MlDemo/Data/trainingdata.tsv");
            var traindata = reader.Read(TrainDataPath);
            

            var est = ctx.Transforms.Conversion.MapValueToKey("CategoryID", "Label")
                .Append(ctx.Transforms.Text.FeaturizeText("Title", "Title_featurized"))
                .Append(ctx.Transforms.Text.FeaturizeText("Description", "Description_featurized"))
                .Append(ctx.Transforms.Concatenate("Features", "Title_featurized", "Description_featurized"))
                .Append(ctx.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            var model = est.Fit(traindata);
            stopwatch.Stop();
            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                ctx.Model.Save(model, fs);
            Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);

            var predictionEngine = model.MakePredictionFunction<ItemData, ItemPrediction>(ctx);

            var testdata = reader.Read(TestDataPath);
            var predictions = model.Transform(testdata);
            var metrics = ctx.MulticlassClassification.Evaluate(predictions, "Label", "Score");

            ConsoleHelper.PrintMultiClassClassificationMetrics("Demo", metrics);

            var prediction = predictionEngine.Predict(new ItemData
            {
                Title = "Sony Blu-Ray Player",
                Description = "Blu-Ray player from Sony, black powers on"
            });

            Console.WriteLine("Predicted catid:{0}", prediction.CategoryID);

        }





        //var mlContext = new MLContext(seed: 0);

        //1.
        //BuildTrainEvaluateAndSaveModel(mlContext);
    }

  
    
}
