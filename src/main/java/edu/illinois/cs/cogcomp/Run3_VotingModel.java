package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Story;
import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.core.utilities.configuration.Configurator;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by snigdha on 12/16/16.
 */
public class Run3_VotingModel extends TwoLevelsClassifierSuperClass {

    public static void main(String args[]) throws Exception {

        /**************** BEGIN MODEL CONFIGURATIONS********************************************************/
        Properties p = new Properties();
        // Input locations
        p.put("datadir","Dataset/RoCStories/");
        p.put("validationFile", p.getProperty("datadir") + "cloze_test_val__spring2016 - cloze_test_ALL_val.tsv");
        p.put("testFile", p.getProperty("datadir") + "test_spring2016.tsv");
        p.put("unannotatedFile", p.getProperty("datadir") + "100KStories.csv");

        p.put("numComponents", "3"); // number of semantic aspects
        p.put("extractFeaturesAgain", Configurator.TRUE); // if set to FALSE then it will read extracted features in aspect0FeaturesTrain,aspect1FeaturesTrain,aspect2FeaturesTrain,aspect0FeaturesTest,aspect1FeaturesTest,aspect2FeaturesTest

        // input for aspect 0: semLM features
        p.put("semLMFeaturesFileTest", "Resources/SemLM_Feats/test_feats.txt");
        p.put("semLMFeaturesFileVal", "Resources/SemLM_Feats/valid_feats.txt");

        // input for aspect 1: sentiment related files
        p.put("sentimentModelDump", "out/sentModel/");
        p.put("retrainSentModel", Configurator.FALSE); // do you want to retrain a sentiment language model from unannotated story corpus? Set to TRUE if out/sentModel/ is empty

        // input for aspect 2: topical Coherence related files
        p.put("gloveVectorsFile", "Resources/preTrainedGloveVectors/glove.6B.100d.txt");
        p.put("stopWordsFile", "Resources/nltkStopwords.txt");

        //  temporary output file: files for storing extracted features (can be reused by other models)
        p.put("aspect0FeaturesTrain", "out/semFeaturesTrain.arff");
        p.put("aspect1FeaturesTrain", "out/sentFeaturesTrain.arff");
        p.put("aspect2FeaturesTrain", "out/topicFeaturesTrain.arff");
        p.put("aspect0FeaturesTest", "out/semFeaturesTest.arff");
        p.put("aspect1FeaturesTest", "out/sentFeaturesTest.arff");
        p.put("aspect2FeaturesTest", "out/topicFeaturesTest.arff");

        // output file: stores predictions
        p.put("predictionsOutputFile_hard", "out/predictions/predictions_HardVoting.csv");
        p.put("predictionsOutputFile_soft", "out/predictions/predictions_SoftVoting.csv");
        /*************************************************************************************************/

        Utils.printInitialMessage(p); // print some initial configuration messages

        Run3_VotingModel model = new Run3_VotingModel();
        model.readCorpus(p);
        if(Boolean.parseBoolean(p.getProperty("extractFeaturesAgain"))) // if set to FALSE then it will read extracted features in "aspect0FeaturesTrain,aspect1FeaturesTrain,aspect2FeaturesTrain,aspect0FeaturesTest,aspect1FeaturesTest,aspect2FeaturesTest"
            model.extractPerComponentFeatures(p);
        model.train(p);
        model.test(p);
    }

    private void train(Properties p){
        trainIndividualAspects_Weka(p);
    }

    private void test(Properties p) throws Exception{
        testWeka(corpus.testStories, "Test", p);
    }

    private void testWeka(ArrayList<Story> stories, String trainOrTest, Properties p) {
        int numComponents = Integer.parseInt(p.getProperty("numComponents"));
        ArrayList<HashMap<String, Prediction>> allAspectsPredictionHms = new ArrayList<>();
        for(int i=0;i<numComponents;i++)
            allAspectsPredictionHms.add(testWekaClassifier(aspectSpecificWekaClassifiers[i], p.getProperty("aspect"+i+"Features"+ trainOrTest))); // reads the predictions of individual aspect classifiers on the Test set

        reportResults(stories, trainOrTest, p, allAspectsPredictionHms);
    }

    private void reportResults(ArrayList<Story> stories, String trainOrTest, Properties p, ArrayList<HashMap<String, Prediction>> allAspectsPredictionHms){
        int numComponents = Integer.parseInt(p.getProperty("numComponents"));
        double correct_voting=0, correct_combined=0;
        Set<String> incorrectIds = new HashSet<>();

        try {
            BufferedWriter bw1 = new BufferedWriter(new FileWriter(p.getProperty("predictionsOutputFile_hard")));
            BufferedWriter bw2 = new BufferedWriter(new FileWriter(p.getProperty("predictionsOutputFile_soft")));
            for (Story s : stories) { // access to corpus needed to get all story ids
                String id = s.instanceId;

                // voting method
                int c1 = 0, c2 = 0;
                for (int i = 0; i < numComponents; i++) {
                    if (allAspectsPredictionHms.get(i).get(id).predictedClass.equals("1"))
                        c1++;
                    else
                        c2++;
                }

                String prediction = getPred(c1, c2);
                String correctness = "";
                if (prediction.equals(String.valueOf(s.answer))) {
                    correctness = "correct";
                    correct_voting++;
                }
                else
                    correctness = "wrong";
                bw1.write(s.instanceId+","+s.answer+","+prediction+","+correctness);
                bw1.newLine();

                // combine prediction from all components
                double probC1 = 0, probC2 = 0;
                for (int i = 0; i < numComponents; i++) {
                    probC1 += Math.log(allAspectsPredictionHms.get(i).get(id).classDist[0]);
                    probC2 += Math.log(allAspectsPredictionHms.get(i).get(id).classDist[1]);

                }

                prediction = getPred(probC1, probC2);
                if (prediction.equals(String.valueOf(s.answer))) {
                    correct_combined++;
                    correctness = "correct";
                }
                else {
                    correctness = "wrong";
                    incorrectIds.add(id);
                }
                bw2.write(s.instanceId+","+s.answer+","+prediction+","+correctness);
                bw2.newLine();
            }
            bw1.close();
            bw2.close();

            System.out.println("\n\nAccuracy of hard voting method on " + trainOrTest + "=" + (correct_voting * 100 / stories.size()));
            System.out.println("Accuracy of soft voting method on " + trainOrTest + "=" + (correct_combined * 100 / stories.size()));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private String getPred(double scoreC1, double scoreC2) {
        if(scoreC1>scoreC2)
            return "1";
        else
            return "2";
    }
}
