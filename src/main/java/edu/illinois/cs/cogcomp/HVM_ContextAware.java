package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by snigdha on 12/26/16.
 */
public class HVM_ContextAware {
    // weights1, weights1: top level weights and features for picking the hidden variable. i.e. choosing an aspect
    // weights2, features2: bottom level weights and features used to pick a label for the instance given aspect
    double GDLearningRate;
    int numGDIter;
    int numAspects;
    double[][] weights1;
    ArrayList<double[]> weights2;

    public HVM_ContextAware(int numAspects, int numAspectClassificationFeatures, ArrayList<Integer> aspectSpecificLabelClassificationFeatDim) {
        this.numAspects = numAspects;
        this.weights1 = new double[numAspects][numAspectClassificationFeatures];
        this.weights2 = new ArrayList<>();
        for(int numFeat: aspectSpecificLabelClassificationFeatDim)
            weights2.add(new double[numFeat]);

        GDLearningRate =0.0001;
        numGDIter = 3000;
    }

    public static class HVModelInstance {
        public String id;
        private int labelMappedTo01;
        public double[] features1 = null;
        public ArrayList<double[]> features2 = null;

        public HVModelInstance(String id, int label, ArrayList<double[]> aspectSpecificFeaturesForLabelClassification, double[] aspectClassificationFeatures) {
            this.id = id;
            this.labelMappedTo01 = label;
            this.features1 = aspectClassificationFeatures;
            features2 = new ArrayList<>();
            for(int i=0;i<aspectSpecificFeaturesForLabelClassification.size();i++)
                features2.add(aspectSpecificFeaturesForLabelClassification.get(i));
        }
    }

    public double[] EM(int numIter, ArrayList<HVModelInstance> instances, ArrayList<HVModelInstance> testInstances, boolean toprint) {
        double[] testAcc = new double[numIter/5 + 2]; //+2 only as buffer
        int counter=0;

        for(int iter=1;iter<numIter+1;iter++) {
            // E Step: compute hidden variable assignments (expectations)
            HashMap<Integer, double[]> hm_instanceId_hvAssign = new HashMap<>();
            for (int instanceInd = 0; instanceInd < instances.size(); instanceInd++) {
                HVModelInstance instance = instances.get(instanceInd);

                // get probability of aspect given this story, and probability of label given this aspect
                double[] aspectProb = new double[numAspects];
                double[] labelProb = new double[numAspects];
                for(int aspectInd=0;aspectInd<numAspects;aspectInd++){
                    aspectProb[aspectInd] = Math.exp((-1) * Utils.dotProduct(weights1[aspectInd],instance.features1));

                    labelProb[aspectInd] = Utils.sigmoid(Utils.dotProduct(weights2.get(aspectInd), instance.features2.get(aspectInd))); // prob of class being 1
                    if(instance.labelMappedTo01==0)
                        labelProb[aspectInd] = 1 - labelProb[aspectInd];
                }
                aspectProb = Utils.normalize(aspectProb);

                // hidden variable assignment: probability of aspect given story and label
                double[] hvAssignments = new double[numAspects];
                for(int aspectInd=0; aspectInd<numAspects;aspectInd++){
                    hvAssignments[aspectInd] = aspectProb[aspectInd] * labelProb[aspectInd];
                }
                hvAssignments = Utils.normalize(hvAssignments);
                hm_instanceId_hvAssign.put(instanceInd,hvAssignments);
            }

            // M Step
            for(int gdIter=0;gdIter<numGDIter;gdIter++) {
                for (int instanceInd = 0; instanceInd < instances.size(); instanceInd++) {
                    HVModelInstance instance = instances.get(instanceInd);

                    // get probabilty of various aspects given story
                    double aspectProb[] = new double[numAspects];
                    for (int aspectInd = 0; aspectInd < numAspects; aspectInd++)
                        aspectProb[aspectInd] = Math.exp((-1)*Utils.dotProduct(weights1[aspectInd], instance.features1));
                    aspectProb = Utils.normalize(aspectProb);

                    // get prob of observation according to various aspects
                    double obs1Prob[] = new double[numAspects];
                    for (int aspectInd = 0; aspectInd < numAspects; aspectInd++) {
                        obs1Prob[aspectInd] = Utils.sigmoid(Utils.dotProduct(weights2.get(aspectInd), instance.features2.get(aspectInd))); // prob of class being 1
                    }

                    for (int aspectInd = 0; aspectInd < numAspects; aspectInd++) {
                        // M Step: update weights1
                        double temp = hm_instanceId_hvAssign.get(instanceInd)[aspectInd]*(1 - aspectProb[aspectInd]);
                        for (int fIndex = 0; fIndex < weights1[aspectInd].length; fIndex++) {
                            weights1[aspectInd][fIndex] -= GDLearningRate *(temp * instance.features1[fIndex] + 0.1*weights1[aspectInd][fIndex]);
                        }

                        // M Step: update weights2
                        double[] aspectWeights = weights2.get(aspectInd);
                        temp = hm_instanceId_hvAssign.get(instanceInd)[aspectInd] * (instance.labelMappedTo01 - obs1Prob[aspectInd]);
                        for (int fIndex = 0; fIndex < instance.features2.get(aspectInd).length; fIndex++) {
                            aspectWeights[fIndex] += GDLearningRate * (temp * instance.features2.get(aspectInd)[fIndex]);// - 0.1*aspectWeights[fIndex]) ;
                        }

                        weights2.remove(aspectInd);
                        weights2.add(aspectInd, aspectWeights);
                    }
                }
            }

            if(iter%5==0) {
                if(toprint) {
                double correct=0;
//                for(HVM_ContextAware.HVModelInstance testInstance: instances)
//                    if(test(testInstance).getFirst())
//                        correct++;
//                double acc1 = correct*100/instances.size();
//                correct=0;
                for(HVM_ContextAware.HVModelInstance testInstance: testInstances)
                    if(test(testInstance).getFirst())
                        correct++;
                double acc = correct*100/testInstances.size();
                testAcc[counter++] = acc;
//                System.out.println("EM iter ="+iter +", Accuracy on train and dev/test set="+acc1+" "+acc);
                    System.out.println("EM iter ="+iter +", Accuracy on dev set="+acc);
                }
                else
                    System.out.println("At iter "+iter +" out of "+numIter);
            }

        }
        return testAcc;
    }

    public Pair<Boolean,String> test(HVModelInstance testInstance){
        return softTest(testInstance);
    }

    private Pair<Boolean,String> softTest(HVModelInstance testInstance){
        String ret = "";

        // get probability of aspect given this story
        double[] aspectProb = new double[numAspects];
        for(int aspectInd=0;aspectInd<numAspects;aspectInd++)
            aspectProb[aspectInd] = Math.exp((-1) * Utils.dotProduct(weights1[aspectInd],testInstance.features1));
        aspectProb = Utils.normalize(aspectProb);

        // get most likely label
        double p1 = 0, p0=0;
        for (int aspectInd = 0; aspectInd < numAspects; aspectInd++) {
            double temp = Utils.sigmoid(Utils.dotProduct(weights2.get(aspectInd), testInstance.features2.get(aspectInd)));
            p1 += aspectProb[aspectInd] * temp; // prob of class being 1
            p0 += aspectProb[aspectInd] * (1-temp);
            ret+= "Probability of aspect number "+(aspectInd+1)+ "=" +aspectProb[aspectInd]+" and prob of option2 acc to this aspect="+temp+"\n";
            ret+= "Contribution of aspect number "+(aspectInd+1)+ " for answer=1 is " + (aspectProb[aspectInd] * (1-temp)) + " and for answer=2 is " + (aspectProb[aspectInd] * temp) + "\n";
            if(testInstance.labelMappedTo01==0)
                ret+= "Contribution of aspect number "+(aspectInd+1)+ " towards correct (not predicted) answer is " + (aspectProb[aspectInd] * (1-temp)) + "\n";
            else
                ret+= "Contribution of aspect number "+(aspectInd+1)+ " towards correct (not predicted) answer is " + (aspectProb[aspectInd] * (temp)) + "\n";
        }
        int pred = 0;
        if(p1>p0)
            pred=1;

        if(pred==testInstance.labelMappedTo01)
            return new Pair(true, ret);
        else
            return new Pair(false, ret);

    }

    private Pair<Boolean,String> hardTest(HVModelInstance testInstance){
        String ret = "";

        // get probability of aspect given this story
        double[] aspectProb = new double[numAspects];
        for(int aspectInd=0;aspectInd<numAspects;aspectInd++)
            aspectProb[aspectInd] = Math.exp((-1) * Utils.dotProduct(weights1[aspectInd],testInstance.features1));
        int mostProbAspectInd = Utils.getMaxEleIndex(aspectProb);
        aspectProb = Utils.normalize(aspectProb);

        // prepare output string for log.txt
        for (int aspectInd = 0; aspectInd < numAspects; aspectInd++) {
            double temp = Utils.sigmoid(Utils.dotProduct(weights2.get(aspectInd), testInstance.features2.get(aspectInd)));
            ret+= "Probability of aspect number "+(aspectInd+1)+ "=" +aspectProb[aspectInd]+" and prob of option2 acc to this aspect="+temp+"\n";
        }

        // get most likely label
        double p1 = Utils.sigmoid(Utils.dotProduct(weights2.get(mostProbAspectInd), testInstance.features2.get(mostProbAspectInd))); // prob of class being 1
        double p0 = 1-p1;

        int pred = 0;
        if(p1>p0)
            pred=1;
        if(pred==testInstance.labelMappedTo01)
            return new Pair(true, ret);
        else
            return new Pair(false, ret);

    }
}
