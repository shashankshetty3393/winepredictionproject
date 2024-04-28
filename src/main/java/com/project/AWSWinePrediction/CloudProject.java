package com.project.AWSWinePreedction;

import org.apache.spark.sql.functions;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.io.IOException;

/**
 * UCID: ss4759
 * Name: Shashank Shetty Manelli
**/
public class CloudProject {
    public static void main(String[] args) {
        String trainingData = "";
        String testData = "";
        String output = "";
        if (args.length > 3) {
            System.err.println("Error occured");
            System.exit(1);
        } else if(args.length ==3){
            trainingData = args[0];
            testData = args[1];
            output = args[2] + "model";
        } else{
            trainingData = "s3://ss4759wine/TrainingDataset.csv";
            testData = "s3://ss4759wine/ValidationDataset.csv";
            output = "s3://ss4759wine/Test.model";
        
        }
        
        SparkSession spark = SparkSession.builder()
        .appName("WineQualityPrediction")
        .config("spark.master", "local")
        .getOrCreate();
        
        JavaSparkContext javaSparkContext = new JavaSparkContext(spark.sparkContext());
        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> trainingdata = spark.read().format("csv")
        .option("header", true)
        .option("quote", "\"") // handle escaped quotes
        .option("delimiter", ";")
        .load(trainingData);
                
        
        Dataset<Row> testdata = spark.read().format("csv")
        .option("header", true)
        .option("quote", "\"") // handle escaped quotes
        .option("delimiter", ";")
        .load(testData);

        String[] inputColumns = {"fixed acidity", "volatile acidity", "citric acid", "chlorides", "total sulfur dioxide", "density", "sulphates", "alcohol"};
        
        for (String col : inputColumns) {
            trainingdata = trainingdata.withColumn(col, trainingdata.col(col).cast("Double")); 
        }
        

        for (String col : inputColumns) {
            testdata = testdata.withColumn(col, testdata.col(col).cast("Double"));
        }
        

        trainingdata = trainingdata.withColumn("quality", trainingdata.col("quality").cast("Double"));
        testdata = testdata.withColumn("quality", testdata.col("quality").cast("Double"));
        
        trainingdata = trainingdata.withColumn("label", functions.when(trainingdata.col("quality").geq(7), 1.0).otherwise(0.0));
        testdata = testdata.withColumn("label", functions.when(testdata.col("quality").geq(7), 1.0).otherwise(0.0));
        

        VectorAssembler assembler = new VectorAssembler()
        .setInputCols(inputColumns)
        .setOutputCol("features");
        
        RandomForestClassifier rf = new RandomForestClassifier()
        .setLabelCol("quality")
        .setFeaturesCol("features")
        .setNumTrees(400)
        .setMaxBins(12)
        .setMaxDepth(22)
        .setSeed(100)
        .setImpurity("entropy");
        
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{assembler, rf});
        
        PipelineModel model = pipeline.fit(trainingdata);
        
        System.out.println(model);
        
        Dataset<Row> predictions = model.transform(testdata);
        
        predictions.select("prediction", "quality", "features").show(5);
        
        MulticlassClassificationEvaluator multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("quality")
        .setPredictionCol("prediction");
        
        System.out.println(multiclassClassificationEvaluator);
        
        double accuracy = multiclassClassificationEvaluator.setMetricName("accuracy").evaluate(predictions);
        mcevaluator.setMetricName("weightedPrecision");
        double precision = multiclassClassificationEvaluator.evaluate(predictions);

        multiclassClassificationEvaluator.setMetricName("weightedRecall");
        double recall = multiclassClassificationEvaluator.evaluate(predictions);

        System.out.println("Accuracy is " + accuracy);
        System.out.println("Precision   is " + precision);
        System.out.println("Recall   is " + recall);

        try {
            model.write().overwrite().save(output);
        } catch (IOException e) {
            System.err.println("Model save failed due to : " + e.getMessage());
        }
    }
}
