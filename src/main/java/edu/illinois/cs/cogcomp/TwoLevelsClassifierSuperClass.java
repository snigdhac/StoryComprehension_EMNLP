package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.Helpers.readData;
import edu.illinois.cs.cogcomp.SemLM.semLMFeatureReader;
import edu.illinois.cs.cogcomp.SentimentLM.sentimentLM_condensed;
import edu.illinois.cs.cogcomp.TopicalCoherence.TopicalCoherenceModel;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Properties;

/**
 * Created by snigdha on 12/16/16.
 */
public class TwoLevelsClassifierSuperClass {
    protected readData corpus = new readData();
    FilteredClassifier[] aspectSpecificWekaClassifiers = null;

    void trainIndividualAspects_Weka(Properties p) {
        // Train individual models
        aspectSpecificWekaClassifiers = new FilteredClassifier[Integer.parseInt(p.getProperty("numComponents"))];
        for(int i = 0; i< aspectSpecificWekaClassifiers.length; i++)
            aspectSpecificWekaClassifiers[i] = trainFilteredClassifier(p.getProperty("aspect"+i+"FeaturesTrain"));
    }

    private FilteredClassifier trainFilteredClassifier(String featuresFile){
        FilteredClassifier fc = null;
        try {
            // read trainIndividualAspects_Weka data
            BufferedReader reader = new BufferedReader(new FileReader(featuresFile));
            Instances trainData = new Instances(reader);
            reader.close();
            trainData.setClassIndex(trainData.numAttributes() - 1); // setting class attribute

            // first attribute is id of the story
            Remove rm = new Remove();
            rm.setAttributeIndices("1");  // remove 1st attributes
            Logistic logisticRegression = new Logistic(); // classifier
            fc = new FilteredClassifier(); // meta-classifier
            fc.setFilter(rm);
            fc.setClassifier(logisticRegression);

            // Train
            fc.buildClassifier(trainData);
        }catch (Exception e){
            e.printStackTrace();
        }
        return fc;
    }

    HashMap<String, Prediction> testWekaClassifier(FilteredClassifier fc, String testFeaturesFile){
        HashMap<String, Prediction> hm = new HashMap<String, Prediction>();
        try {
            // read trainIndividualAspects_Weka and test data
            BufferedReader reader = new BufferedReader(new FileReader(testFeaturesFile));
            Instances testData = new Instances(reader);
            reader.close();
            testData.setClassIndex(testData.numAttributes() - 1); // setting class attribute

            // Make predictions
            for (int i = 0; i < testData.numInstances(); i++) {
                Instance testIns = testData.instance(i);
                String prediction = testData.classAttribute().value((int) fc.classifyInstance(testIns));
                hm.put(testData.instance(i).stringValue(testData.attribute(0)), new Prediction(prediction,fc.distributionForInstance(testIns)));
            }
        }catch (Exception e){
            e.printStackTrace();
        }
        return hm;
    }

    protected void extractPerComponentFeatures(Properties p) {
        // Aspect 1: Read Semantic LM features
        System.out.println("Reading semLM features");
        HashMap<String, double[]> hm_validSemLMFeatures= semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("semLMFeaturesFileVal"));
        Utils.writeFeaturesToArffFile(corpus.valStories, hm_validSemLMFeatures, p.getProperty("aspect0FeaturesTrain"));
        HashMap<String, double[]> hm_testSemLMFeatures= semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("semLMFeaturesFileTest"));
        Utils.writeFeaturesToArffFile(corpus.testStories, hm_testSemLMFeatures, p.getProperty("aspect0FeaturesTest"));

        // Aspect 2: extract the sentiment LM features
        System.out.println("Extracting sentimentLM features");
        System.out.println("Uncomment these lines in TwoLevelClassifiersSuperClass.java here");
        sentimentLM_condensed sentFeatExtractor = new sentimentLM_condensed();
        sentFeatExtractor.getLM(corpus.unannotatedStories, Boolean.parseBoolean(p.getProperty("retrainSentModel")), p.getProperty("sentimentModelDump"));
        HashMap<String, double[]> hm_validSentFeatures = sentFeatExtractor.extractFeaturesForStories(corpus.valStories);
        Utils.writeFeaturesToArffFile(corpus.valStories, hm_validSentFeatures, p.getProperty("aspect1FeaturesTrain"));
        HashMap<String, double[]> hm_testSentFeatures = sentFeatExtractor.extractFeaturesForStories(corpus.testStories);
        Utils.writeFeaturesToArffFile(corpus.testStories, hm_testSentFeatures, p.getProperty("aspect1FeaturesTest"));

        // Aspect 3: extract the topical coherence features
        System.out.println("Extracting topical coherence features");
        TopicalCoherenceModel topicalModel = new TopicalCoherenceModel(p.getProperty("stopWordsFile"), p.getProperty("gloveVectorsFile"));
        HashMap<String, double[]> hm_validTopicFeatures= topicalModel.extractTopicFeatures(corpus.valStories);
        Utils.writeFeaturesToArffFile(corpus.valStories, hm_validTopicFeatures, p.getProperty("aspect2FeaturesTrain"));
        HashMap<String, double[]> hm_testTopicFeatures = topicalModel.extractTopicFeatures(corpus.testStories);
        Utils.writeFeaturesToArffFile(corpus.testStories, hm_testTopicFeatures, p.getProperty("aspect2FeaturesTest"));

        if(Integer.parseInt(p.getProperty("numComponents"))> 3) {
            // Aspect 4: Read Entity LM features
            System.out.println("Reading entityLM features");
            HashMap<String, double[]> hm_validEntityLMFeatures = semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("entityLMFeaturesFileVal"));
            HashMap<String, double[]> hm_testEntityLMFeatures = semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("entityLMFeaturesFileTest"));
            Utils.writeFeaturesToArffFile(corpus.valStories, hm_validEntityLMFeatures, p.getProperty("aspect3FeaturesTrain"));
            Utils.writeFeaturesToArffFile(corpus.testStories, hm_testEntityLMFeatures, p.getProperty("aspect3FeaturesTest"));
        }
    }

    protected void readCorpus(Properties p) {
        // Read the corpus
        corpus.readAllData(p.getProperty("validationFile"), p.getProperty("testFile"));
        corpus.readUnannotatedData(p.getProperty("unannotatedFile"));
    }


    public class Prediction{
        public String predictedClass = null;
        public double[] classDist = null;
        public Prediction(String s, double[] c){
            predictedClass = s;
            classDist = c;
        }
    }
}
