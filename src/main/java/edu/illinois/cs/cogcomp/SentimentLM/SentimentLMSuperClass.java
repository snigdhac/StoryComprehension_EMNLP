package edu.illinois.cs.cogcomp.SentimentLM;

import edu.illinois.cs.cogcomp.SentHandler.SentimentHandler;
import edu.illinois.cs.cogcomp.annotation.AnnotatorException;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;

import java.util.HashMap;

/**
 * Created by Snigdha on 12/1/16.
 */
public class SentimentLMSuperClass {
    SentimentHandler sentimentHandler = new SentimentHandler();

    protected int[] getSentiment(String sent1) throws AnnotatorException {
        return sentimentHandler.getSentiment(sent1);
    }

    public int categorize(int i) {
        if(i>=1)
            return 1;
        else if(i<=-1)
            return -1;
        else
            return 0;
    }

    public  String intArrToString(int[] arr, char delimiter, int end) {
        String ret = "";
        for(int i=0;i<end;i++){
            ret+=""+arr[i]+delimiter;
        }
        return ret.substring(0, ret.length()-1);
    }

    protected double getProb(int happiness, Pair<Double, Double> val) {
        if(val!=null){
            if(happiness>0)
                return val.getFirst();
            if(happiness<0)
                return val.getSecond();
        }
        else
            System.err.println("Couldn't find value corresponding to the key "+happiness+"! I'll return 0");
        return 0;
    }

    public  HashMap<String,Pair<Double,Double>> updatePairHM(HashMap<String, Pair<Double, Double>> hm4gram, String seq4gram, int sentimentLastSent) {
        Pair<Double, Double> val = new Pair<>(0.0,0.0);
        if(hm4gram.containsKey(seq4gram))
            val = hm4gram.get(seq4gram);
        if(sentimentLastSent ==1){
            val.setFirst(val.getFirst()+1);
        }
        if(sentimentLastSent == -1){
            val.setSecond(val.getSecond()+1);
        }
        hm4gram.put(seq4gram,val);
        return hm4gram;
    }

    public  HashMap<String, Pair<Double, Double>> normalizePairHM(HashMap<String, Pair<Double, Double>> hm2gram) {
        for(String key: hm2gram.keySet()){
            Pair<Double,Double> p = hm2gram.get(key);
            Double sum = p.getFirst()+ p.getSecond();
            if(sum!=0) {
                p.setFirst(p.getFirst() / sum);
                p.setSecond(p.getSecond() / sum);
            }
            hm2gram.put(key, p);
        }
        return hm2gram;
    }
}
