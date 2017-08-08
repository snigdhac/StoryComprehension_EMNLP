package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Story;
import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.annotation.AnnotatorException;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.Random;

/**
 * Created by snigdha on 12/26/16.
 */
public class Run4_ContextAwareHVModel extends JointModelSuperClass {
    static int optimalNumberOfIter = 30; // For the story completion dataset.

    public static void main(String args[]) throws Exception {

        Run4_ContextAwareHVModel model = new Run4_ContextAwareHVModel();
        Properties p = model.getProperties();

        Utils.printInitialMessage(p); // print some initial configuration messages

        model.readCorpus(p);

        if(Boolean.parseBoolean(p.getProperty("extractFeaturesAgain"))) // If false then reads the outputtrainFile and outputtestFile directly
            model.extractFeatures(p);
        else
            model.readFeatures(p);

        ArrayList<HVM_ContextAware.HVModelInstance> trainInstances = model.formatInstances(model.corpus.valStories,
                model.hm_validSemLMFeatures, model.hm_validSentFeatures, model.hm_validTopicFeatures);
        ArrayList<HVM_ContextAware.HVModelInstance> testInstances = model.formatInstances(model.corpus.testStories,
                model.hm_testSemLMFeatures, model.hm_testSentFeatures, model.hm_testTopicFeatures);

        ArrayList<Integer> featDim = new ArrayList<>();
        for(int i=0;i<trainInstances.get(0).features2.size();i++)
            featDim.add(trainInstances.get(0).features2.get(i).length);

        int numAspectClassificationFeatures  =trainInstances.get(0).features1.length;

        // Parameter Selection
        optimalNumberOfIter = parameterSelection(trainInstances, p, numAspectClassificationFeatures, featDim);

        // Train the Hidden Variable Model
        System.out.println("\nStarting final training and testing mode");
        HVM_ContextAware classifier = new HVM_ContextAware(Integer.parseInt(p.getProperty("numComponents")), numAspectClassificationFeatures, featDim);
        classifier.EM(optimalNumberOfIter, trainInstances, testInstances, false);

        // printing the test results
        double correct=0;
        System.out.println("Outputing results to "+p.getProperty("predictionsOutputFile")+". Format is story id, gold answer, our prediction, correct?");

        BufferedWriter bw = new BufferedWriter(new FileWriter(p.getProperty("predictionsOutputFile")));

        for(HVM_ContextAware.HVModelInstance testInstance: testInstances){
            Pair<Boolean, String> pair = classifier.test(testInstance); // test instance
            if(pair.getFirst())
                correct++;

            int prediction = 0;
            String correctness = "";
            Story story = getStory(model.corpus.testStories, testInstance.id);
            if(pair.getFirst()) {
                prediction = story.answer;
                correctness = "correct";
            }
            else {
                correctness = "wrong";
                if(story.answer==2)
                    prediction = 1;
                if(story.answer==1)
                    prediction = 2;
            }
            bw.write(story.instanceId+","+story.answer+","+prediction+","+correctness+"\n");
        }
        bw.close();
        System.out.println("Accuracy on test set = "+ (correct*100/testInstances.size()));
    }

    private static Story getStory(ArrayList<Story> testStories, String id) {
        for(Story story: testStories)
            if(story.instanceId.equals(id))
                return story;
        return null;
    }

    private ArrayList<HVM_ContextAware.HVModelInstance> formatInstances(ArrayList<Story> valStories, HashMap<String, double[]> hm1,
                                                                        HashMap<String, double[]> hm2,
                                                                        HashMap<String, double[]> hm3) throws AnnotatorException {
        ArrayList<HVM_ContextAware.HVModelInstance> instances = new ArrayList<>();
        int[] numFeats;
        numFeats = new int[3];
        numFeats[0] = hm1.get(hm1.keySet().iterator().next()).length;
        numFeats[1] = hm2.get(hm2.keySet().iterator().next()).length;
        numFeats[2] = hm3.get(hm3.keySet().iterator().next()).length;

        for(Story story:valStories){
            int label = 0;
            if(story.answer==2)
                label = 1;
            ArrayList<double[]> allFeat = new ArrayList<>();
            double semLMFeats[] = Utils.getFeatures(hm1, story.instanceId, numFeats[0]);
            double sentFeats[] = Utils.getFeatures(hm2, story.instanceId, numFeats[1]);
            double topicFeats[] = Utils.getFeatures(hm3, story.instanceId, numFeats[2]);
            allFeat.add(semLMFeats);
            allFeat.add(sentFeats);
            allFeat.add(topicFeats);

            // extract features for aspect classification
            double[] aspectClassificationFeatures = extractAspectClassificationFeatures(topicFeats, semLMFeats, sentFeats);
            HVM_ContextAware.HVModelInstance instance = new HVM_ContextAware.HVModelInstance(story.instanceId, label, allFeat, aspectClassificationFeatures);
            instances.add(instance);
        }
        return instances;
    }

    private double[] extractAspectClassificationFeatures(double[] topicFeatures, double[] semLMFeat, double[] sentFeat)
    throws AnnotatorException {


        double ret[];
        ret = new double[1+1+1+1]; // one feature for each aspect and a bias term

        ret[0] = Math.abs(topicFeatures[0]-topicFeatures[1]);

        // sentiment based features:
        // If numFeaturesTYPES = d. format is <feat for o1, feat for o2> repreated d times and the d comparativeBinaryFeat
        int numsentFeats = sentFeat.length;
        if(numsentFeats%3!=0){
            System.err.println("WARNING. expected 3*d sentiment features");
        }
        int d = numsentFeats/3;
        double diff = 0;
        for(int i=0;i<2*d;i=i+2)
            diff += Math.abs(sentFeat[i] - sentFeat[i+1]);
        ret[1] = diff/d;

        //semLM based features
        // If numFeaturesTYPES = d. format is <feat for o1, feat for o2, comparativeBinaryFeat> repreated d times
        if(semLMFeat.length%3!=0){
            System.err.println("WARNING. expected 3*d semLM features");
        }
        d = semLMFeat.length/3;
        diff = 0;
        for(int i=0;i<semLMFeat.length;i=i+3) {
            diff = Math.abs(semLMFeat[i] - semLMFeat[i + 1]);
        }
        ret[2] = diff/d;

        ret[3] = 1; //bias term

        return ret;
    }

    private static int parameterSelection(ArrayList<HVM_ContextAware.HVModelInstance> trainInstances, Properties p, int numAspectClassificationFeatures, ArrayList<Integer> featDim){
        // parameter selection mode
        int numIter = 100, numDevSets=5;
        double[] avgDevAcc = new double[numIter/5 + 2]; // length should be same as that of the array returned by HVM_ContextAware.EM()
        System.out.println("In Parameter selection mode. Selecting optimal number of EM iterations by considering average accuracy on "+numDevSets+" held out dev sets (20%) from training set");
        Random rand = new Random(4065);
        for(int set=0;set<numDevSets;set++) {
            System.out.println("**********Running set "+set);
            ArrayList<HVM_ContextAware.HVModelInstance> tempTrain = new ArrayList<>();
            ArrayList<HVM_ContextAware.HVModelInstance> tempDev = new ArrayList<>();
            for(HVM_ContextAware.HVModelInstance trainInstance: trainInstances){
                if(rand.nextDouble()>0.8)
                    tempDev.add(trainInstance);
                else
                    tempTrain.add(trainInstance);
            }
            HVM_ContextAware tempClassifier = new HVM_ContextAware(Integer.parseInt(p.getProperty("numComponents")), numAspectClassificationFeatures, featDim);
            double acc[] = tempClassifier.EM(numIter, tempTrain, tempDev, true);

            for(int i=0;i<acc.length;i++)
                avgDevAcc[i] += acc[i];
        }
        for(int i=0;i<avgDevAcc.length;i++)
            avgDevAcc[i] = avgDevAcc[i]/numDevSets;
        int optimalNumIter = Utils.getMaxIndex(avgDevAcc);

        System.out.println("Printing average development set accuracies");
        System.out.println(Utils.printArr(avgDevAcc));
        System.out.println("Optimal number of iterations corresponds to the "+(optimalNumIter+1)+" entry in the above array. Optimal num iter ="+(optimalNumIter+1)*5);
        optimalNumIter = (optimalNumIter+1)*5; // 5 because saving results after every 5 iter
        System.out.println("Optimal Number of iterations="+optimalNumIter);
        return optimalNumIter;
    }
}
