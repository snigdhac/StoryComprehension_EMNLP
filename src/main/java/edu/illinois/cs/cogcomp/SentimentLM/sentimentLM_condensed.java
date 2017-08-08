package edu.illinois.cs.cogcomp.SentimentLM;

import edu.illinois.cs.cogcomp.Helpers.Story;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Constructs a 'Climax Sentiment Modeler' from 5 sentences long stories.
 * It considers sentiment of condensed sentences. Sentence 1 is start, Sentences 2 and 3 form Body, Sentence 4 forms Pre-climax, and Sentence 5 forms climax.
 * It models P(climax sentiment=1|previous sentiment trajectory), and P(climax sentiment=-1|previous sentiment trajectory)
 * Then extracts features based on the above probabilities for the validation and the test set.
 * Created by Snigdha on 11/2/16.
 */

public class sentimentLM_condensed extends SentimentLMSuperClass {
    private HashMap<String, Pair<Double, Double>> hm3gramMPQA = new HashMap<String, Pair<Double, Double>>();
    private HashMap<String, Pair<Double, Double>> hm2gramMPQA = new HashMap<String, Pair<Double, Double>>();
    private HashMap<String, Pair<Double, Double>> hm1gramMPQA = new HashMap<String, Pair<Double, Double>>();
    private HashMap<String, Pair<Double, Double>> hmcollapsedMPQA = new HashMap<String, Pair<Double, Double>>();

    private HashMap<String, Pair<Double, Double>> hm3gramBingLiu = new HashMap<String, Pair<Double, Double>>();
    private HashMap<String, Pair<Double, Double>> hm2gramBingLiu = new HashMap<String, Pair<Double, Double>>();
    private HashMap<String, Pair<Double, Double>> hm1gramBingLiu = new HashMap<String, Pair<Double, Double>>();
    private HashMap<String, Pair<Double, Double>> hmcollapsedBingLiu = new HashMap<String, Pair<Double, Double>>();

    public void getLM(ArrayList<Story> unannotatedStories, boolean retrainSentModel, String storedFilesLoc){
        try {
            if (retrainSentModel) {
                System.out.println("Learning sentiment LM from unannotated set");
                learnLM(unannotatedStories);
                System.out.println("Dumping all probabilities learned for Sentiment model at " + storedFilesLoc);
                dumpLearnedHMs(storedFilesLoc);
            } else {
                System.out.println("Reading Sentiment Language models from " + storedFilesLoc);
                readStoredLM(storedFilesLoc);
            }
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void learnLM(ArrayList<Story> stories) throws Exception{
        int totalInstances =0;
        for (Story story: stories){
            totalInstances++;
            if (totalInstances % 1000 == 0) {
                System.out.println(totalInstances + "/" + stories.size());
            }
            int[] sentimentBingLiu = new int[5];
            for (int i=0;i<sentimentBingLiu.length;i++)
                sentimentBingLiu[i] = 0;
            int[] sentimentMPQA = new int[5];
            for (int i=0;i<sentimentMPQA.length;i++)
                sentimentMPQA[i] = 0;

            for (int i = 0; i < 5; i++) {
                int[] allSentiments = null;
                if (i<4) {
                    String sent = story.first4Sentences[i];
                    allSentiments = getSentiment(sent);
                }
                if (i==4) {
                    if (story.answer==1)
                        allSentiments = getSentiment(story.option1);
                    else if (story.answer==2)
                        allSentiments = getSentiment(story.option2);
                    else
                        System.err.println("story's answer is "+story.answer+". I expect 1 or 2. Ignoring this instance");
                }

                sentimentBingLiu[i] = allSentiments[0]-allSentiments[1]; //num positive words - num negative words
                sentimentMPQA[i] = allSentiments[2]-allSentiments[3]; //num positive words - num negative words
            }
            String seqCollapsedBingLiu = ""+categorize(sentimentBingLiu[0] + sentimentBingLiu[1] + sentimentBingLiu[2] + sentimentBingLiu[3]);
            String seqCollapsedMPQA = ""+categorize(sentimentMPQA[0] + sentimentMPQA[1] + sentimentMPQA[2] + sentimentMPQA[3]);

            int[] seqCondensedBingLiu = new int[3];
            seqCondensedBingLiu[0] = categorize(sentimentBingLiu[0]);
            seqCondensedBingLiu[1] = categorize(sentimentBingLiu[1] + sentimentBingLiu[2]);
            seqCondensedBingLiu[2] = categorize(sentimentBingLiu[3]);

            int[] seqCondensedMPQA = new int[3];
            seqCondensedMPQA[0] = categorize(sentimentMPQA[0]);
            seqCondensedMPQA[1] = categorize(sentimentMPQA[1] + sentimentMPQA[2]);
            seqCondensedMPQA[2] = categorize(sentimentMPQA[3]);

            hm3gramBingLiu = updatePairHM(hm3gramBingLiu, intArrToString(seqCondensedBingLiu,':',3), categorize(sentimentBingLiu[4]));
            hm2gramBingLiu = updatePairHM(hm2gramBingLiu, intArrToString(seqCondensedBingLiu,':',2), categorize(sentimentBingLiu[4]));
            hm1gramBingLiu = updatePairHM(hm1gramBingLiu, intArrToString(seqCondensedBingLiu,':',1), categorize(sentimentBingLiu[4]));
            hmcollapsedBingLiu = updatePairHM(hmcollapsedBingLiu, seqCollapsedBingLiu, categorize(sentimentBingLiu[4]));

            hm3gramMPQA = updatePairHM(hm3gramMPQA, intArrToString(seqCondensedMPQA,':',3), categorize(sentimentMPQA[4]));
            hm2gramMPQA = updatePairHM(hm2gramMPQA, intArrToString(seqCondensedMPQA,':',2), categorize(sentimentMPQA[4]));
            hm1gramMPQA = updatePairHM(hm1gramMPQA, intArrToString(seqCondensedMPQA,':',1), categorize(sentimentMPQA[4]));
            hmcollapsedMPQA = updatePairHM(hmcollapsedMPQA, seqCollapsedMPQA, categorize(sentimentMPQA[4]));
        }

        this.normalizeAllHMs();
    }

    public void dumpLearnedHMs(String dir){
        dumpHM(hm3gramMPQA, dir+"hm3gramMPQA.txt");
        dumpHM(hm2gramMPQA, dir+"hm2gramMPQA.txt");
        dumpHM(hm1gramMPQA, dir+"hm1gramMPQA.txt");
        dumpHM(hmcollapsedMPQA, dir+"hmcollapsedMPQA.txt");

        dumpHM(hm3gramBingLiu, dir+"hm3gramBingLiu.txt");
        dumpHM(hm2gramBingLiu, dir+"hm2gramBingLiu.txt");
        dumpHM(hm1gramBingLiu, dir+"hm1gramBingLiu.txt");
        dumpHM(hmcollapsedBingLiu, dir+"hmcollapsedBingLiu.txt");

    }

    private void dumpHM(HashMap<String, Pair<Double, Double>> hm, String filename){
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
            for (String key : hm.keySet()) {
                bw.write(key +"\t"+hm.get(key).getFirst()+"\t"+hm.get(key).getSecond());
                bw.newLine();
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void readStoredLM(String dir){
        hm3gramMPQA = readHM(dir+"hm3gramMPQA.txt");
        hm2gramMPQA = readHM(dir+"hm2gramMPQA.txt");
        hm1gramMPQA = readHM(dir+"hm1gramMPQA.txt");
        hmcollapsedMPQA = readHM(dir+"hmcollapsedMPQA.txt");

        hm3gramBingLiu = readHM(dir+"hm3gramBingLiu.txt");
        hm2gramBingLiu = readHM(dir+"hm2gramBingLiu.txt");
        hm1gramBingLiu = readHM(dir+"hm1gramBingLiu.txt");
        hmcollapsedBingLiu = readHM(dir+"hmcollapsedBingLiu.txt");
    }

    private HashMap<String,Pair<Double,Double>> readHM(String s) {
        HashMap<String,Pair<Double,Double>> hm = new HashMap<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(s));
            String line = "";
            while ((line = br.readLine()) != null) {
                String toks[] = line.split("\t");
                hm.put(toks[0].trim(), new Pair(Double.parseDouble(toks[1].trim()), Double.parseDouble(toks[2].trim())));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if(hm.keySet().size()==0){
            System.err.println("Could not read "+s);
            System.err.println("Highly recommended to fix this as it will effect the quality of sentiment based features.");
        }
        return hm;
    }


    private void extractFeatures(ArrayList<Story> stories, String outfile) throws Exception{
        BufferedWriter bw = new BufferedWriter(new FileWriter(outfile));
        bw.write("@RELATION storycompletion\n" +
                "\n" +
                "@ATTRIBUTE option1_3gramProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option2_3gramProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option1_2gramProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option2_2gramProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option1_1gramProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option2_1gramProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option1_CollapsedSeqProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option2_CollapsedSeqProb_BingLiu  NUMERIC\n" +
                "@ATTRIBUTE option1_3gramProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option2_3gramProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option1_2gramProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option2_2gramProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option1_1gramProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option2_1gramProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option1_CollapsedSeqProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE option2_CollapsedSeqProb_MPQA  NUMERIC\n" +
                "@ATTRIBUTE class        {1,2}\n" +
                "\n" +
                "@DATA\n");

        int totalInstances=0;
        for (Story story: stories){
            totalInstances++;
            if (totalInstances % 100 == 0)
                System.out.println(totalInstances + "/" + stories.size());

            double[] features = extractFeaturesFromStory(story);

            //write features
            for(int i=0;i<features.length;i++)
                bw.write(features[i]+",");
            bw.write(story.answer+"");
            bw.newLine();
        }
        bw.close();
    }

    public HashMap<String, double[]> extractFeaturesForStories(ArrayList<Story> stories) {
        int totalInstances = 0;
        HashMap<String, double[]> hm = new HashMap<>();
        try {
            for (Story story : stories) {
                totalInstances++;
                if(totalInstances%100==0)
                    System.out.println("Extracted Sentiment features for " + totalInstances + "/" + stories.size() + "stories");
                hm.put(story.instanceId, extractFeaturesFromStory(story));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return hm;
    }

    public double[] extractFeaturesFromStory(Story story) throws Exception{
        if(isLMEmpty()){
            System.err.println("Language Model is empty. No meaningful features will be extracted");
        }
        int[] sentimentBingLiu = new int[4];
        for(int i=0;i<4;i++)
            sentimentBingLiu[i] = 0;
        int[] sentimentMPQA = new int[4];
        for(int i=0;i<4;i++)
            sentimentMPQA[i] = 0;

        for (int i = 0; i < 4; i++) {
            int[] allSentiments = null;
            String sent = story.first4Sentences[i];
            allSentiments = getSentiment(sent);

            sentimentBingLiu[i] = allSentiments[0]-allSentiments[1]; //num positive words - num negative words
            sentimentMPQA[i] = allSentiments[2]-allSentiments[3]; //num positive words - num negative words
        }
        String seqCollapsedBingLiu = ""+categorize(sentimentBingLiu[0] + sentimentBingLiu[1] + sentimentBingLiu[2] + sentimentBingLiu[3]);
        String seqCollapsedMPQA = ""+categorize(sentimentMPQA[0] + sentimentMPQA[1] + sentimentMPQA[2] + sentimentMPQA[3]);

        int[] seqCondensedBingLiu = new int[3];
        seqCondensedBingLiu[0] = categorize(sentimentBingLiu[0]);
        seqCondensedBingLiu[1] = categorize(sentimentBingLiu[1] + sentimentBingLiu[2]);
        seqCondensedBingLiu[2] = categorize(sentimentBingLiu[3]);

        int[] seqCondensedMPQA = new int[3];
        seqCondensedMPQA[0] = categorize(sentimentMPQA[0]);
        seqCondensedMPQA[1] = categorize(sentimentMPQA[1] + sentimentMPQA[2]);
        seqCondensedMPQA[2] = categorize(sentimentMPQA[3]);

        int[] option1AllSentiments = getSentiment(story.option1);
        int[] option2AllSentiments = getSentiment(story.option2);

        int happiness1BingLiu = option1AllSentiments[0] - option1AllSentiments[1]; // happiness is not -1, 0, 1
        int happiness2BingLiu = option2AllSentiments[0] - option2AllSentiments[1];// happiness is not -1, 0, 1
        int happiness1MPQA = option1AllSentiments[2] - option1AllSentiments[3]; // happiness is not -1, 0, 1
        int happiness2MPQA = option2AllSentiments[2] - option2AllSentiments[3];// happiness is not -1, 0, 1

        // get features
        double[] features = new double[24];
        for(int i=0;i<features.length;i++)
            features[i] = 0;
        features[0] = getProb(happiness1BingLiu, hm3gramBingLiu.get(intArrToString(seqCondensedBingLiu,':',3)));// happiness is not -1, 0, 1
        features[1] = getProb(happiness2BingLiu, hm3gramBingLiu.get(intArrToString(seqCondensedBingLiu,':',3)));// happiness is not -1, 0, 1
        features[2] = getProb(happiness1BingLiu, hm2gramBingLiu.get(intArrToString(seqCondensedBingLiu,':',2)));// happiness is not -1, 0, 1
        features[3] = getProb(happiness2BingLiu, hm2gramBingLiu.get(intArrToString(seqCondensedBingLiu,':',2)));// happiness is not -1, 0, 1
        features[4] = getProb(happiness1BingLiu, hm1gramBingLiu.get(intArrToString(seqCondensedBingLiu,':',1)));// happiness is not -1, 0, 1
        features[5] = getProb(happiness2BingLiu, hm1gramBingLiu.get(intArrToString(seqCondensedBingLiu,':',1)));// happiness is not -1, 0, 1
        features[6] = getProb(happiness1BingLiu, hmcollapsedBingLiu.get(seqCollapsedBingLiu));
        features[7] = getProb(happiness2BingLiu, hmcollapsedBingLiu.get(seqCollapsedBingLiu));

        features[8] = getProb(happiness1MPQA, hm3gramMPQA.get(intArrToString(seqCondensedMPQA,':',3)));
        features[9] = getProb(happiness2MPQA, hm3gramMPQA.get(intArrToString(seqCondensedMPQA,':',3)));
        features[10] = getProb(happiness1MPQA, hm2gramMPQA.get(intArrToString(seqCondensedMPQA,':',2)));
        features[11] = getProb(happiness2MPQA, hm2gramMPQA.get(intArrToString(seqCondensedMPQA,':',2)));
        features[12] = getProb(happiness1MPQA, hm1gramMPQA.get(intArrToString(seqCondensedMPQA,':',1)));
        features[13] = getProb(happiness2MPQA, hm1gramMPQA.get(intArrToString(seqCondensedMPQA,':',1)));
        features[14] = getProb(happiness1MPQA, hmcollapsedMPQA.get(seqCollapsedMPQA));
        features[15] = getProb(happiness2MPQA, hmcollapsedMPQA.get(seqCollapsedMPQA));

        // binary features (like Haoruo)
        features[16] = (features[1]>features[0]) ? 1 : -1;
        features[17] = (features[3]>features[2]) ? 1 : -1;
        features[18] = (features[5]>features[4]) ? 1 : -1;
        features[19] = (features[7]>features[6]) ? 1 : -1;
        features[20] = (features[9]>features[8]) ? 1 : -1;
        features[21] = (features[11]>features[10]) ? 1 : -1;
        features[22] = (features[13]>features[12]) ? 1 : -1;
        features[23] = (features[15]>features[14]) ? 1 : -1;

        return features;
    }

    private boolean isLMEmpty() {
        if(hm3gramBingLiu.keySet().size()==0 || hm2gramBingLiu.keySet().size()==0 || hm1gramBingLiu.keySet().size()==0 )
            return true;
        if(hm3gramMPQA.keySet().size()==0 || hm2gramMPQA.keySet().size()==0 || hm1gramMPQA.keySet().size()==0)
            return true;
        if(hmcollapsedBingLiu.keySet().size()==0 || hmcollapsedMPQA.keySet().size()==0)
            return true;
        return false;
    }

    private void normalizeAllHMs() {
        hm3gramBingLiu = normalizePairHM(hm3gramBingLiu);
        hm2gramBingLiu = normalizePairHM(hm2gramBingLiu);
        hm1gramBingLiu = normalizePairHM(hm1gramBingLiu);
        hmcollapsedBingLiu = normalizePairHM(hmcollapsedBingLiu);

        hm3gramMPQA = normalizePairHM(hm3gramMPQA);
        hm2gramMPQA = normalizePairHM(hm2gramMPQA);
        hm1gramMPQA = normalizePairHM(hm1gramMPQA);
        hmcollapsedMPQA = normalizePairHM(hmcollapsedMPQA);
    }

}
