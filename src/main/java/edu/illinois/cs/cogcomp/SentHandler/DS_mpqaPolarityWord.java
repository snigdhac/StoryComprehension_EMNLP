package edu.illinois.cs.cogcomp.SentHandler;

/**
 * Created by Snigdha on 8/7/17.
 */
public class DS_mpqaPolarityWord {

    //    String subjectivity, len;
    String word, pos;
    Boolean stemmed;


    public DS_mpqaPolarityWord(String word, String pos, boolean stemmed) {
        this.word = word;
        this.pos = pos;
        this.stemmed = stemmed;
    }

    @Override
    public int hashCode()
    {
        return (word.hashCode() + pos.hashCode() + stemmed.hashCode());
    }

    @Override
    public boolean equals(Object o1)
    {   DS_mpqaPolarityWord o = (DS_mpqaPolarityWord) o1;
        return (word.equals(o.word) && pos.equals(o.pos) && stemmed==o.stemmed);
    }
}
