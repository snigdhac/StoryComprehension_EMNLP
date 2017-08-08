package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Story;
import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.core.utilities.configuration.Configurator;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

import java.io.*;
import java.util.*;

/**
 * Created by snigdha on 12/16/16.
 */
public class Run2_DisJointModel extends TwoLevelsClassifierSuperClass {
    FilteredClassifier level2WekaClassifier = null;

    public static void main(String args[]) throws Exception{

        /**************** BEGIN MODEL CONFIGURATIONS********************************************************/
        Properties p = new Properties();

        p.put("numComponents","3"); // number of semantic aspects
        p.put("extractFeaturesAgain", Configurator.TRUE); // if set to FALSE then it will read extracted features in aspect0FeaturesTrain,aspect1FeaturesTrain,aspect2FeaturesTrain,aspect0FeaturesTest,aspect1FeaturesTest,aspect2FeaturesTest

        // Input locations
        p.put("datadir","Dataset/RoCStories/");
        p.put("validationFile", p.getProperty("datadir") + "cloze_test_val__spring2016 - cloze_test_ALL_val.tsv");
        p.put("testFile", p.getProperty("datadir") + "test_spring2016.tsv");
        p.put("unannotatedFile", p.getProperty("datadir") + "100KStories.csv");

        // input for aspect 0: semLM features
        p.put("semLMFeaturesFileTest","Resources/SemLM_Feats/test_feats.txt");
        p.put("semLMFeaturesFileVal", "Resources/SemLM_Feats/valid_feats.txt");

        // input for aspect 1: sentiment related files
        p.put("sentimentModelDump" ,"out/sentModel/");
        p.put("retrainSentModel", Configurator.FALSE); // do you want to retrain a sentiment language model from unannotated story corpus? Set to TRUE if out/sentModel/ is empty

        // input for aspect 2: topical Coherence related files
        p.put("gloveVectorsFile" , "Resources/preTrainedGloveVectors/glove.6B.100d.txt");
        p.put("stopWordsFile", "Resources/nltkStopwords.txt");

        // temporary output file: files for storing extracted features (can be reused by other models)
        p.put("aspect0FeaturesTrain", "out/semFeaturesTrain.arff");
        p.put("aspect1FeaturesTrain", "out/sentFeaturesTrain.arff");
        p.put("aspect2FeaturesTrain", "out/topicFeaturesTrain.arff");
        p.put("aspect0FeaturesTest", "out/semFeaturesTest.arff");
        p.put("aspect1FeaturesTest", "out/sentFeaturesTest.arff");
        p.put("aspect2FeaturesTest", "out/topicFeaturesTest.arff");

        // temporary output file: final training and testing files
        p.put("level2TrainingFile","out/2LevelDisjointModelTrainingFile.arff");
        p.put("level2TestingFile", "out/2LevelDisjointModelTestingFile.arff");

        // output file: stores predictions
        p.put("predictionsOutputFile", "out/predictions/predictions_AspectAwareSoftEnsemble.csv");
        /*************************************************************************************************/

        Utils.printInitialMessage(p); // print some initial configuration messages

        Run2_DisJointModel model = new Run2_DisJointModel();
        model.readCorpus(p);
        model.extractAllFeatures(p);
        model.trainAndEvaluate(p);

        // delete all files created files
        (new File(p.getProperty("level2TestingFile"))).delete();
        (new File(p.getProperty("level2TrainingFile"))).delete();
    }

    private void extractAllFeatures(Properties p) {
        if(Boolean.parseBoolean(p.getProperty("extractFeaturesAgain")))
            extractPerComponentFeatures(p); // extracts features for 3 components (Level 1 classifiers)

        trainIndividualAspects_Weka(p); // trainIndividualComponents

        extractFeaturesForSecondLevelClassifier(p, corpus.valStories, p.getProperty("level2TrainingFile"),"Train"); // extract features for level 2 classifier
        extractFeaturesForSecondLevelClassifier(p, corpus.testStories, p.getProperty("level2TestingFile"),"Test"); // extract features for level 2 classifier

    }

    private Set<String> trainAndEvaluate(Properties p) {
        return trainAndEvaluate_Weka(p.getProperty("level2TrainingFile"), p.getProperty("level2TestingFile"), p.getProperty("predictionsOutputFile"));
    }

    private Set<String> trainAndEvaluate_Weka(String validationFeaturesFile, String testFeaturesFile, String predictionsOutputFile) {
        Set<String> incorrectIds = new HashSet<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(validationFeaturesFile));
            Instances trainData = new Instances(reader);
            reader.close();
            trainData.setClassIndex(trainData.numAttributes() - 1); // setting class attribute

            reader = new BufferedReader(new FileReader(testFeaturesFile));
            Instances testData = new Instances(reader);
            reader.close();
            testData.setClassIndex(testData.numAttributes() - 1); // setting class attribute

            Remove rm = new Remove();
            rm.setAttributeIndices("1");  // remove 1st attribute
            Logistic logisticRegression = new Logistic(); // classifier
            level2WekaClassifier = new FilteredClassifier(); // meta-classifier
            level2WekaClassifier.setFilter(rm);
            level2WekaClassifier.setClassifier(logisticRegression);

            // Train
            level2WekaClassifier.buildClassifier(trainData);

            // Make predictions
            BufferedWriter bw = new BufferedWriter(new FileWriter(predictionsOutputFile));
            bw.write("InputStoryid,Actual,Predicted,Correctness");
            bw.newLine();
            String correctness = "";
            int correct=0;
            for (int i = 0; i < testData.numInstances(); i++) {
                double pred = level2WekaClassifier.classifyInstance(testData.instance(i));
                String actual = testData.classAttribute().value((int) testData.instance(i).classValue());
                String predicted = testData.classAttribute().value((int) pred);

                if(predicted.equals(actual)) {
                    correct++;
                    correctness = "correct";
                }
                else{
                    incorrectIds.add(testData.instance(i).stringValue(testData.attribute(0)));
                    correctness = "wrong";
                }
                bw.write(testData.instance(i).stringValue(testData.attribute(0))+","+actual+","+predicted+","+correctness);
                bw.newLine();
            }
            System.out.println("\n\nAccuracy on test set= "+ ((double)correct*100/testData.numInstances()));
            bw.close();

        }catch (Exception e){
            e.printStackTrace();
        }
        return incorrectIds;
    }

    private void extractFeaturesForSecondLevelClassifier(Properties p, ArrayList<Story> stories, String outfile, String TrainOrTest){
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(outfile));

            int numComponents = Integer.parseInt(p.getProperty("numComponents"));
            bw.write(getHardHeader(numComponents));

            ArrayList<HashMap<String, Prediction>> allAspectsPredictionHms = new ArrayList<>();
            for (int i = 0; i < numComponents; i++)
                allAspectsPredictionHms.add(testWekaClassifier(aspectSpecificWekaClassifiers[i], p.getProperty("aspect" + i + "Features"+TrainOrTest))); // reads the predictions of individual aspect classifiers on the Test set

            for (Story s : stories) { // access to corpus needed to get all story ids
                String id = s.instanceId;
                bw.write("\"" + id + "\",");
                for(int i=0;i<numComponents;i++){
                    double[] classDist = allAspectsPredictionHms.get(i).get(id).classDist;
                    // for prediction as features
                    int prediction =0;
                    if(classDist[0]>classDist[1])
                        prediction = 1;
                    if(classDist[1]>classDist[0])
                        prediction = 2;
                    bw.write(prediction+",");
                }
                bw.write(s.answer + "");
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String getHardHeader(int numComponents) {
        String header = "@RELATION storycompletion\n\n";
        header += "@ATTRIBUTE storyId  STRING\n";
        for(int i=0;i<numComponents;i++) {
            header += "@ATTRIBUTE featFrom" + i + "  NUMERIC\n";
        }
        header +="@ATTRIBUTE class        {1,2}\n\n@DATA\n";
        return header;
    }
}
