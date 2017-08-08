package edu.illinois.cs.cogcomp.Helpers;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Properties;
import java.util.Set;

/**
 * Created by snigdha on 12/16/16.
 */
public class Utils {

    private static String getHeader(int numFeats) throws Exception {
        String header = "@RELATION storycompletion\n\n";
        header += "@ATTRIBUTE storyId  STRING\n";
        for(int i=0;i<numFeats;i++)
            header += "@ATTRIBUTE featureNo"+i+"  NUMERIC\n";
        header +="@ATTRIBUTE class        {1,2}\n\n@DATA\n";
        return header;
    }

    public static void writeFeaturesToArffFile(ArrayList<Story> stories, HashMap<String, double[]> featureHM, String outfile) {
        try {
            int numFeats = featureHM.get(featureHM.keySet().iterator().next()).length;

            BufferedWriter bw = new BufferedWriter(new FileWriter(outfile));
            bw.write(getHeader(numFeats));
            for (Story story : stories) {
                String storyId = story.instanceId;
                double[] features = featureHM.get(storyId);

                // if features for this story are absent then replace them with all zeros
                if (features == null) {
                    features = new double[numFeats];
                    for (int i = 0; i < features.length; i++)
                        features[i] = 0.0;
                }

                //write features
                bw.write("\"" + storyId + "\",");
                for (int i = 0; i < features.length; i++)
                    bw.write(features[i] + ",");
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

    public static double[] getFeatures(HashMap<String, double[]> hm, String storyId, int numFeat) {
        double[] features = hm.get(storyId);
        if (features == null) {
            features = new double[numFeat];
            for (int i = 0; i < features.length; i++)
                features[i] = 0.0;
        }
        return features;
    }

    public static void dumpIncorrectPred(ArrayList<Story> testStories, Set<String> incorrectTestIds, String logFile) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(logFile));
            for (Story s : testStories) {
                if (incorrectTestIds.contains(s.instanceId)) {
                    bw.write(s.pprintStory());
                    bw.newLine();
                }
            }
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Double maxElement(double[] arr) {
        if(arr.length==0)
            return null;
        Double max = arr[0];
        for(int i=1;i<arr.length;i++)
            if(arr[i]>max)
                max = arr[i];
        return max;
    }

    public static String printArr(double[] arr){
        String ret ="";
        for(double a: arr)
            ret += a+" ";
        return ret;
    }

    public static double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public static double dotProduct(double[] w, double[] x){
        if(w.length!=x.length) {
            System.err.println("Lengths of the two vectors should be same for dot product. Returning -1000;");
            return -1000;
        }
        double sum = 0;
        for(int i=0;i<w.length;i++)
            sum += w[i]*x[i];
        return sum;
    }

    public static double[] normalize(double[] arr) {
        double sum = 0;
        for(double a: arr)
            sum+=a;
        for(int i=0;i<arr.length;i++)
            if(arr[i]!=0)
                arr[i] /=sum;
        return arr;
    }

    public static int getMaxEleIndex(double[] arr) {
        if(arr.length==0)
            return -1;
        int ret = 0;
        double max = arr[0];
        for(int i=1;i<arr.length;i++) {
            if (max < arr[i]) {
                ret = i;
                max = arr[i];
            }
        }
        return ret;
    }

    public static int getMaxIndex(double[] arr) {
        double max = -999;
        int maxIndex = -1;
        for(int i=0;i<arr.length;i++){
            if(arr[i]>max){
                max = arr[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void printInitialMessage(Properties p) {
        System.out.println("Dataset used = " + p.getProperty("datadir"));
        System.out.println("Number of semantic aspects used = "+p.getProperty("numComponents"));

        if(!Boolean.parseBoolean(p.getProperty("retrainSentModel")))
            System.out.println("Lazy mode: I recycled the sentiment LM learned from RoC 100K Stories dataset. Use 'p.put(\"retrainSentModel\", Configurator.TRUE);' if you want to learn a new sentiment LM from a different dataset or you get (relevant) file not found error.");
        else
            System.out.println("Learning a new sentiment LM from the unannotatedFile that you specified in main() (Check beginning of your main function if you don't understand this).");

        if(!Boolean.parseBoolean(p.getProperty("extractFeaturesAgain")))
            System.out.println("Lazy mode: Not extracting features for this model. Reading them from the temporary output files that you specified in main() (Check beginning of your main function if you don't understand this). Use 'p.put(\"extractFeaturesAgain\", Configurator.TRUE);' if you want to extract fresh features or you get (relevant) file not found error.");
        else
            System.out.println("Extracting features");
        System.out.println();
    }
}
