package edu.illinois.cs.cogcomp.SemLM;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 * Created by snigdha on 12/16/16.
 */
public class semLMFeatureReader {
    public static HashMap<String, double[]> readSemLMClassificationFeatures(String semLMFeaturesFile){
        HashMap<String, double[]> hm_storyId_semLMFeat = new HashMap<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(semLMFeaturesFile));
            String line = "";
            while((line=br.readLine())!=null){
                //remove trailing comma, if any
                if(line.endsWith(","))
                    line = line.substring(0,line.length()-1);

                String [] toks = line.trim().split("\t");
                String[] stringFeats = toks[1].split(",");
                double[] features = new double[stringFeats.length];
                for(int i=0;i<stringFeats.length;i++)
                    features[i] = Double.parseDouble(stringFeats[i]);
                hm_storyId_semLMFeat.put(toks[0], features);
            }
            br.close();
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return hm_storyId_semLMFeat;
    }
}
