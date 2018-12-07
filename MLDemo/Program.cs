using System;
using System.IO;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System.Diagnostics;
using Microsoft.ML.Legacy.Transforms;

namespace MLDemo
{
    class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string BaseDatasetsLocation = @"../../../../Data";
        private static string TrainDataPath = $"./Data/trainingdata.tsv";
        private static string TrainData10kPath = $"./Data/trainingdata10k.tsv";
        private static string TrainData500kPath = $"./Data/trainingdata500k.tsv";

        private static string TestDataPath = $"./Data/testingdata.tsv";
        private static string BaseModelsPath = @"../../../MLModels";
        private static string ModelPath = $"./Data/ItemCategorizationModel.zip";
        private static string Model10kPath = $"./Data/ItemCategorizationModel10k.zip";
        private static string Model500kPath = $"./Data/ItemCategorizationModel500k.zip";

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
                    //new TextLoader.Column("Description", DataKind.Text, 2),
                    }
            });

            //var traindata = reader.Read("/users/ryansmith/Projects/mldemo/MlDemo/Data/trainingdata.tsv");
            var traindata = reader.Read(TrainDataPath);
            var traindata10k = reader.Read(TrainData10kPath);
            var traindata500k = reader.Read(TrainData500kPath);

            var est = ctx.Transforms.Conversion.MapValueToKey("CategoryID", "Label")

                //.Append(ctx.Transforms.Text.FeaturizeText("Title", "Title_featurized"))
                .Append(ctx.Transforms.Text.TokenizeWords("Title","Title_tokenized"))
                .Append(ctx.Transforms.Text.RemoveStopWords("Title_tokenized","Title_cleaned"))
                .Append(ctx.Transforms.Text.FeaturizeText("Title_cleaned","Title_featurized"))
                //.Append(ctx.Transforms.Text.RemoveStopWords("Description","DescriptionCleaned"))
                //.Append(ctx.Transforms.Text.FeaturizeText("Description", "Description_featurized"))
                //.Append(ctx.Transforms.Text.TokenizeWords("Description", "Description_tokenized"))
                //.Append(ctx.Transforms.Text.RemoveStopWords("Description_tokenized", "Description_cleaned"))
                //.Append(ctx.Transforms.Text.FeaturizeText("Description_cleaned", "Description_featurized"))
                .Append(ctx.Transforms.Concatenate("Features", "Title_featurized"))
                .Append(ctx.MulticlassClassification.Trainers.StochasticDualCoordinateAscent("Label", "Features"))
                .Append(ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Stopwatch stopwatch = new Stopwatch();
            stopwatch.Start();
            var model = est.Fit(traindata);
            stopwatch.Stop();
            using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                ctx.Model.Save(model, fs);
            Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);

            stopwatch.Reset();
            stopwatch.Start();
            var model10k = est.Fit(traindata10k);
            stopwatch.Stop();
            using (var fs = new FileStream(Model10kPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                ctx.Model.Save(model10k, fs);
            Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);

            stopwatch.Reset();
            stopwatch.Start();
            var model500k = est.Fit(traindata500k);
            stopwatch.Stop();
            using (var fs = new FileStream(Model500kPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                ctx.Model.Save(model500k, fs);
            Console.WriteLine("Time elapsed: {0}", stopwatch.Elapsed);

            var predictionEngine = model.MakePredictionFunction<ItemData, ItemPrediction>(ctx);
            var predictionEngine10k = model10k.MakePredictionFunction<ItemData, ItemPrediction>(ctx);

            var predictionEngine500k = model500k.MakePredictionFunction<ItemData, ItemPrediction>(ctx);


            var testdata = reader.Read(TestDataPath);
            var predictions = model.Transform(testdata);
            var predictions10k = model10k.Transform(testdata);
            var predictions500k = model500k.Transform(testdata);

            var metrics = ctx.MulticlassClassification.Evaluate(predictions, "Label", "Score");
            var metrics10k = ctx.MulticlassClassification.Evaluate(predictions10k, "Label", "Score");

            var metrics500k = ctx.MulticlassClassification.Evaluate(predictions500k, "Label", "Score");

            ConsoleHelper.PrintMultiClassClassificationMetrics("Demo", metrics);
            ConsoleHelper.PrintMultiClassClassificationMetrics("Demo10k", metrics10k);

            ConsoleHelper.PrintMultiClassClassificationMetrics("Demo500k", metrics500k);

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
