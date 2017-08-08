package edu.illinois.cs.cogcomp.Helpers;

/**
 * Created by Snigdha on 11/2/16.
 */
public class Story {
    public String instanceId;
    public String option1;
    public String option2;
    public String[] first4Sentences;
    public int answer;

    public Story(String instanceId, String sent1, String sent2, String sent3, String sent4, String option1, String option2, int answer) {
        this.instanceId = instanceId;
        first4Sentences = new String[]{sent1, sent2, sent3, sent4};
        this.option1 = option1;
        this.option2 = option2;
        this.answer = answer;
    }

    public String pprintStory(){
        String ret = "";
        for (String sent : first4Sentences)
            ret+=sent + "\n";
        ret+=option1 + "\n";
        ret+=option2 + "\n";
        ret+="Correct answer = "+answer + "\n";
        return ret;
    }

    public String first4Sents(){
        String ret = "";
        for(String s:first4Sentences)
            ret+=s+" ";
        return ret.trim();
    }
}
