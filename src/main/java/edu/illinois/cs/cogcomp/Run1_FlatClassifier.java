package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Story;
import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.Helpers.readData;
import edu.illinois.cs.cogcomp.SemLM.semLMFeatureReader;
import edu.illinois.cs.cogcomp.SentimentLM.sentimentLM_condensed;
import edu.illinois.cs.cogcomp.TopicalCoherence.TopicalCoherenceModel;
import edu.illinois.cs.cogcomp.core.utilities.configuration.Configurator;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.util.*;

/**
 * Created by snigdha on 12/22/16.
 */
public class Run1_FlatClassifier {
    protected readData corpus = new readData();

    public static void main(String args[]) throws Exception {

        /**************** BEGIN MODEL CONFIGURATIONS********************************************************/
        Properties p = new Properties();

        p.put("numComponents","3"); // number of semantic aspects
        p.put("extractFeaturesAgain", Configurator.TRUE); // If false then reads the outputtrainFile and outputtestFile directly

        // Input locations
        p.put("datadir","Dataset/RoCStories/");
        p.put("validationFile", p.getProperty("datadir") + "cloze_test_val__spring2016 - cloze_test_ALL_val.tsv");
        p.put("testFile", p.getProperty("datadir") + "test_spring2016.tsv");
        p.put("unannotatedFile", p.getProperty("datadir") + "100KStories.csv");

        // input for aspect 0: semLM features
        p.put("semLMFeaturesFileTest", "Resources/SemLM_Feats/test_feats.txt");
        p.put("semLMFeaturesFileVal", "Resources/SemLM_Feats/valid_feats.txt");

        // input for aspect 1: sentiment related files
        p.put("sentimentModelDump", "out/sentModel/");
        p.put("retrainSentModel", Configurator.FALSE); // do you want to retrain a sentiment language model from unannotated story corpus? Set to TRUE if out/sentModel/ is empty

        // input for aspect 2: topical Coherence related files
        p.put("gloveVectorsFile", "Resources/preTrainedGloveVectors/glove.6B.100d.txt");
        p.put("stopWordsFile", "Resources/nltkStopwords.txt");

        // temporary output file: stores the train and test sets for LR model
        p.put("outputtrainFile", "out/validation_sentLM_semLM_topical.arff");
        p.put("outputtestFile", "out/test_sentLM_semLM_topical.arff");

        // output file: stores predictions
        p.put("predictionsOutputFile", "out/predictions/predictions_LR.csv");
        /*************************************************************************************************/


        /**** Start *****/
        Utils.printInitialMessage(p); // print some initial configuration messages

        Run1_FlatClassifier model = new Run1_FlatClassifier();
        model.readCorpus(p);

        System.out.println("No. of validation and test stories="+model.corpus.valStories.size()+" and "+model.corpus.testStories.size());

        if(Boolean.parseBoolean(p.getProperty("extractFeaturesAgain"))) // If false then reads the outputtrainFile and outputtestFile directly
            model.extractFeatures(p); // extracts features for 3 components

        model.trainAndEvaluateWeka(p.getProperty("outputtrainFile"), p.getProperty("outputtestFile"), p.getProperty("predictionsOutputFile"));

        // delete the created files
//        (new File(p.getProperty("outputtrainFile"))).delete();
//        (new File(p.getProperty("outputtestFile"))).delete();

    }

    private void extractFeatures(Properties p) {
        // Aspect 1: Read Semantic LM features
        System.out.println("Reading semLM features");
        HashMap<String, double[]> hm_validSemLMFeatures= semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("semLMFeaturesFileVal"));
        HashMap<String, double[]> hm_testSemLMFeatures= semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("semLMFeaturesFileTest"));

        // Aspect 2: extract the sentiment LM features
        System.out.println("Extracting sentimentLM features");
        sentimentLM_condensed sentFeatExtractor = new sentimentLM_condensed();
        sentFeatExtractor.getLM(corpus.unannotatedStories, Boolean.parseBoolean(p.getProperty("retrainSentModel")), p.getProperty("sentimentModelDump"));
        HashMap<String, double[]> hm_validSentFeatures = sentFeatExtractor.extractFeaturesForStories(corpus.valStories);
        HashMap<String, double[]> hm_testSentFeatures = sentFeatExtractor.extractFeaturesForStories(corpus.testStories);

        // Aspect 3: extract the topical coherence features
        System.out.println("Extracting topical coherence features");
        TopicalCoherenceModel topicalModel = new TopicalCoherenceModel(p.getProperty("stopWordsFile"), p.getProperty("gloveVectorsFile"));
        HashMap<String, double[]> hm_validTopicFeatures= topicalModel.extractTopicFeatures(corpus.valStories);
        HashMap<String, double[]> hm_testTopicFeatures = topicalModel.extractTopicFeatures(corpus.testStories);

        writeToArffFile(p.getProperty("outputtrainFile"), corpus.valStories, hm_validSemLMFeatures, hm_validSentFeatures, hm_validTopicFeatures, null);
        writeToArffFile(p.getProperty("outputtestFile"), corpus.testStories, hm_testSemLMFeatures, hm_testSentFeatures, hm_testTopicFeatures, null);

    }

    private Set<String> trainAndEvaluateWeka(String validationFeaturesFile, String testFeaturesFile, String predictionsOutputFile) {
        Set<String> incorrectIds = new HashSet<>();
        try {
            // read trainIndividualAspects_Weka and test data
            BufferedReader reader = new BufferedReader(new FileReader(validationFeaturesFile));
            Instances trainData = new Instances(reader);
            reader.close();
            trainData.setClassIndex(trainData.numAttributes() - 1); // setting class attribute

            reader = new BufferedReader(new FileReader(testFeaturesFile));
            Instances testData = new Instances(reader);
            reader.close();
            testData.setClassIndex(testData.numAttributes() - 1); // setting class attribute

            // first attribute is id of the story
            Remove rm = new Remove();
            rm.setAttributeIndices("1");  // remove 1st attribute
            Logistic logisticRegression = new Logistic(); // classifier
            FilteredClassifier fc = new FilteredClassifier(); // meta-classifier
            fc.setFilter(rm);
            fc.setClassifier(logisticRegression);

            // Train
            fc.buildClassifier(trainData);

            // Make predictions
            BufferedWriter bw = new BufferedWriter(new FileWriter(predictionsOutputFile));
            bw.write("InputStoryid,Actual,Predicted,Correctness");
            bw.newLine();
            int correct=0;
            for (int i = 0; i < testData.numInstances(); i++) {
                double pred = fc.classifyInstance(testData.instance(i));
                String actual = testData.classAttribute().value((int) testData.instance(i).classValue());
                String predicted = testData.classAttribute().value((int) pred);
                String correctness = "";
                if(predicted.equals(actual)) {
                    correct++;
                    correctness = "correct";
                }
                else {// incorrect prediction
                    incorrectIds.add(testData.instance(i).stringValue(testData.attribute(0)));
                    correctness = "wrong";
                }
                bw.write(testData.instance(i).stringValue(testData.attribute(0))+","+actual+","+predicted+","+correctness);
                bw.newLine();
            }
            bw.close();
            System.out.println("\n\nAccuracy on test set= "+ ((double)correct*100/testData.numInstances()));

        }catch (Exception e){
            e.printStackTrace();
        }
        return incorrectIds;
    }

    private void writeToArffFile(String outfile, ArrayList<Story> stories, HashMap<String, double[]> hm_SemLMFeatures,
                                 HashMap<String, double[]> hm_SentFeatures, HashMap<String, double[]> hm_TopicFeatures,
                                 HashMap<String, double[]> hm_EntityLMFeatures) {
        try {
            int[] numFeats = null;
            if(hm_EntityLMFeatures!=null)
                numFeats = new int[4];
            else
                numFeats = new int[3];
            numFeats[0] = hm_SemLMFeatures.get(hm_SemLMFeatures.keySet().iterator().next()).length;
            numFeats[1] = hm_SentFeatures.get(hm_SentFeatures.keySet().iterator().next()).length;
            numFeats[2] = hm_TopicFeatures.get(hm_TopicFeatures.keySet().iterator().next()).length;
            if(hm_EntityLMFeatures!=null)
                numFeats[3] = hm_EntityLMFeatures.get(hm_EntityLMFeatures.keySet().iterator().next()).length;

            BufferedWriter bw = new BufferedWriter(new FileWriter(outfile));
            bw.write(getHeader(numFeats));
            for (Story story : stories) {
                String storyId = story.instanceId;
                bw.write("\"" + storyId + "\",");

                double[] features = Utils.getFeatures(hm_SemLMFeatures, storyId, numFeats[0]);
                for (int i = 0; i < features.length; i++)
                    bw.write(features[i] + ",");
                features = Utils.getFeatures(hm_SentFeatures, storyId, numFeats[1]);
                for (int i = 0; i < features.length; i++)
                    bw.write(features[i] + ",");
                features = Utils.getFeatures(hm_TopicFeatures, storyId, numFeats[2]);
                for (int i = 0; i < features.length; i++)
                    bw.write(features[i] + ",");
                if(hm_EntityLMFeatures!=null) {
                    features = Utils.getFeatures(hm_EntityLMFeatures, storyId, numFeats[3]);
                    for (int i = 0; i < features.length; i++)
                        bw.write(features[i] + ",");
                }
                bw.write(story.answer + "");
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private String getHeader(int[] numFeats) throws Exception {
        String header = "@RELATION storycompletion\n\n";
        header += "@ATTRIBUTE storyId  STRING\n";
        for(int n=0;n<numFeats.length;n++)
            for(int i=0;i<numFeats[n];i++)
                header += "@ATTRIBUTE featureNo"+n+"_"+i+"  NUMERIC\n";
        header +="@ATTRIBUTE class        {1,2}\n\n@DATA\n";
        return header;
    }

    protected void readCorpus(Properties p) {
        // Read the corpus
        corpus.readAllData(p.getProperty("validationFile"), p.getProperty("testFile"));
        corpus.readUnannotatedData(p.getProperty("unannotatedFile"));
    }
}
