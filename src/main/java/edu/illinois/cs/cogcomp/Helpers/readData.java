package edu.illinois.cs.cogcomp.Helpers;

import com.opencsv.CSVReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

/**
 * Created by Snigdha on 11/2/16.
 */
public class readData {
    public ArrayList<Story> valStories = null;
    public ArrayList<Story> testStories = null;
    public ArrayList<Story> unannotatedStories = null;

    public void readAllData(String validationFile, String testFile){
        valStories = readAnnotatedFile(validationFile);
        testStories = readAnnotatedFile(testFile);
    }

    static ArrayList<Story> readAnnotatedFile(String dataFile){
        ArrayList<Story> ret = new ArrayList<Story>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(dataFile));
            String line = br.readLine(); // read Header
            while ((line = br.readLine()) != null) {
                String toks[] = line.split("\t");
                if (toks.length != 8) {
                    System.err.println("I expected 8 columns in the following line:");
                    System.err.println(line);
                }

                String instanceId = toks[0];
                String sent1 = toks[1];
                String sent2 = toks[2];
                String sent3 = toks[3];
                String sent4 = toks[4];
                String option1 = toks[5];
                String option2 = toks[6];
                int answer = Integer.parseInt(toks[7]);

                Story story = new Story(instanceId, sent1, sent2, sent3, sent4, option1, option2, answer);
                ret.add(story);
            }
            br.close();
        } catch(Exception e){
            e.printStackTrace();
        }

        if(ret.size()==0){
            System.err.println("Didn't read any stories from "+dataFile);
        }
        return ret;
    }

    public void readUnannotatedData(String unannotatedFile) {

        unannotatedStories = new ArrayList<>();
        try {
            CSVReader reader = new CSVReader(new FileReader(unannotatedFile));
            String[] toks = reader.readNext(); // read Header
            while ((toks = reader.readNext()) != null) {
                if (toks.length != 7) {
                    System.err.println("I expected 7 columns");
                }

                String instanceId = toks[0];
                String sent1 = toks[2];
                String sent2 = toks[3];
                String sent3 = toks[4];
                String sent4 = toks[5];
                String option1 = toks[6];
                String option2 = null;
                int answer = 1;

                Story story = new Story(instanceId, sent1, sent2, sent3, sent4, option1, option2, answer);
                unannotatedStories.add(story);
            }
        } catch(Exception e){
            e.printStackTrace();
        }

        if(unannotatedStories.size()==0){
            System.err.println("Didn't read any stories from "+unannotatedStories);
        }
    }
}
