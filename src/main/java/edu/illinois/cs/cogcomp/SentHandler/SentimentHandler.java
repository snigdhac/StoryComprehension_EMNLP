package edu.illinois.cs.cogcomp.SentHandler;

import edu.illinois.cs.cogcomp.annotation.AnnotatorException;
import edu.illinois.cs.cogcomp.annotation.AnnotatorService;
import edu.illinois.cs.cogcomp.core.datastructures.Pair;
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.Constituent;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.Relation;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.TextAnnotation;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.View;
import edu.illinois.cs.cogcomp.core.utilities.configuration.Configurator;
import edu.illinois.cs.cogcomp.core.utilities.configuration.ResourceManager;
import edu.illinois.cs.cogcomp.nlp.common.PipelineConfigurator;
import edu.illinois.cs.cogcomp.nlp.pipeline.IllinoisPipelineFactory;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;

/**
 * Created by Snigdha on 8/7/17.
 */
public class SentimentHandler {
    protected AnnotatorService annotator = null;
    protected HashSet<String> negativeWordsBingLiu = null;
    protected HashSet<String> positiveWordsBingLiu = null;
    protected HashMap<DS_mpqaPolarityWord, String> hm_word_polarityMpqa = null;

    private boolean considerNeg = true;
    private String negativePolarityWordsFile = "Resources/BingLiuList/negative-words.txt";
    private String positivePolarityWordsFile = "Resources/BingLiuList/positive-words.txt";
    private String mpqaPolarityFile = "Resources/subjectivity_clues_hltemnlp05_MPQA/subjclueslen1-HLTEMNLP05.tff";

    public SentimentHandler(){
        initializeAnnotator();
        readPolarityFiles();
    }

    public SentimentHandler(boolean considerNegation, String negativeBingLiuPolarityWordsFile, String positiveBingLiuPolarityWordsFile, String mpqaPolarityFile){
        initializeAnnotator();
        this.considerNeg = considerNegation;
        this.negativePolarityWordsFile = negativeBingLiuPolarityWordsFile;
        this.positivePolarityWordsFile = positiveBingLiuPolarityWordsFile;
        this.mpqaPolarityFile = mpqaPolarityFile;
        readPolarityFiles();
    }

    public void initializeAnnotator() {
        Properties nonDefaultProp = new Properties();
        nonDefaultProp.put(PipelineConfigurator.USE_POS.key, Configurator.TRUE);
        nonDefaultProp.put(PipelineConfigurator.USE_LEMMA.key, Configurator.TRUE);
        nonDefaultProp.put(PipelineConfigurator.USE_NER_CONLL.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_NER_ONTONOTES.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_SHALLOW_PARSE.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_STANFORD_DEP.key, Configurator.TRUE);
        nonDefaultProp.put(PipelineConfigurator.USE_STANFORD_PARSE.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_SRL_VERB.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_SRL_NOM.key, Configurator.FALSE);

        try {
            // Create the AnnotatorService object
            this.annotator = IllinoisPipelineFactory.buildPipeline(new ResourceManager(nonDefaultProp));
        }catch(Exception e){
            System.err.println("There was an exception with the Annotator Service. Note annotator points to null and will create problems later with text processing.");
            e.printStackTrace();
        }
    }

    public Pair<Integer, Integer> getHappiness(String sent) throws AnnotatorException {
        int[] sentimentDegree = getSentiment(sent);
        int happinessBingLiu = categorize(sentimentDegree[0] - sentimentDegree[1]);
        int happinessMPQA = categorize(sentimentDegree[2] - sentimentDegree[3]);
        return new Pair<>(happinessBingLiu, happinessMPQA);
    }

    public int[] getSentiment(String sent) throws AnnotatorException {
        int numBingLiuPosWords = 0, numBingLiuNegWords = 0, numMpqaNegWords = 0, numMpqaPosWords = 0;
        int numBingLiuPosWordsVBRBJJ = 0, numBingLiuNegWordsVBRBJJ = 0, numMpqaNegWordsVBRBJJ = 0, numMpqaPosWordsVBRBJJ = 0;
        TextAnnotation ta = this.annotator.createBasicTextAnnotation("1", "1", sent);
        annotator.addView(ta, ViewNames.LEMMA);
        annotator.addView(ta, ViewNames.POS);

        View lemmView = ta.getView(ViewNames.LEMMA);
        View posView = ta.getView(ViewNames.POS);
        View depView = null; // the dependency parse is needed to determine if this word is negated

        for (Constituent wordLemm : lemmView) {
            String wordSurfaceForm = wordLemm.getSurfaceForm();
            String posCode = posView.getConstituentsCovering(wordLemm).get(0).getLabel();

            String polarityBingLiu = "", polarityMpqa = "";

            // check BingLiu polarity
            if (negativeWordsBingLiu.contains(wordSurfaceForm))
                polarityBingLiu = "negative";
            if (positiveWordsBingLiu.contains(wordSurfaceForm))
                polarityBingLiu = "positive";

            // check MPQA polarity
            polarityMpqa = getMpqaPolarity(posCode, wordSurfaceForm, wordLemm);

            // is this word negated
            boolean isNeg = false;
            try {
                if (considerNeg && (polarityBingLiu.length() != 0 || polarityMpqa.length() != 0)) {
                    if (depView == null) {
                        annotator.addView(ta, ViewNames.DEPENDENCY_STANFORD);
                        depView = ta.getView(ViewNames.DEPENDENCY_STANFORD);
                    }
                    isNeg = isNegated(wordLemm, depView);
                }
            }catch (Exception e){
                e.printStackTrace();
            }

            // reverse polarity if negated
            if(isNeg) {
                if(polarityBingLiu.equals("positive"))
                    polarityBingLiu="negative";
                else if(polarityBingLiu.equals("negative"))
                    polarityBingLiu="positive";
                if(polarityMpqa.equals("positive"))
                    polarityMpqa="negative";
                else if(polarityMpqa.equals("negative"))
                    polarityMpqa="positive";
            }

            if (polarityBingLiu.equals("negative"))
                numBingLiuNegWords++;
            if (polarityBingLiu.equals("positive"))
                numBingLiuPosWords++;

            if (polarityMpqa.equals("negative"))
                numMpqaNegWords++;
            if (polarityMpqa.equals("positive"))
                numMpqaPosWords++;

            // consider only verbs, adverbs, and adjectives
            if((posCode.startsWith("VB") || posCode.startsWith("JJ") || posCode.startsWith("RB"))){
                if (polarityBingLiu.equals("negative"))
                    numBingLiuNegWordsVBRBJJ++;
                if (polarityBingLiu.equals("positive"))
                    numBingLiuPosWordsVBRBJJ++;

                if (polarityMpqa.equals("negative"))
                    numMpqaNegWordsVBRBJJ++;
                if (polarityMpqa.equals("positive"))
                    numMpqaPosWordsVBRBJJ++;
            }

        }

        return new int[]{numBingLiuPosWords, numBingLiuNegWords, numMpqaPosWords, numMpqaNegWords, numBingLiuPosWordsVBRBJJ, numBingLiuNegWordsVBRBJJ, numMpqaPosWordsVBRBJJ, numMpqaNegWordsVBRBJJ};
    }

    private String getMpqaPolarity(String posCode, String wordSurfaceForm, Constituent wordLemm) {
        String polarityMpqa = null;
        String pos ="";
        if(posCode.startsWith("VB"))
            pos = "verb";
        if(posCode.startsWith("NN"))
            pos = "noun";
        if(posCode.startsWith("JJ"))
            pos = "adj";
        if(posCode.startsWith("RB"))
            pos = "adverb";
        boolean stemmed = false;
        DS_mpqaPolarityWord w = new DS_mpqaPolarityWord(wordSurfaceForm, pos, stemmed);
        polarityMpqa = hm_word_polarityMpqa.get(w);
        if(polarityMpqa==null) {
            w = new DS_mpqaPolarityWord(wordSurfaceForm, "anypos", stemmed);
            polarityMpqa = hm_word_polarityMpqa.get(w);
        }
        stemmed = true;
        if(polarityMpqa==null) {
            w = new DS_mpqaPolarityWord(wordLemm.getLabel(), pos, stemmed);
            polarityMpqa = hm_word_polarityMpqa.get(w);
        }
        if(polarityMpqa==null) {
            w = new DS_mpqaPolarityWord(wordLemm.getLabel(), "anypos", stemmed);
            polarityMpqa = hm_word_polarityMpqa.get(w);
        }
        if(polarityMpqa==null)
            return "";
        else
            return polarityMpqa;
    }

    private void readPolarityFiles(){
        readPolarityLexiconBingLiu(negativePolarityWordsFile, positivePolarityWordsFile);
        this.hm_word_polarityMpqa = readMpqaPolarityLexicon(mpqaPolarityFile);
    }

    private void readPolarityLexiconBingLiu(String negativePolarityWordsFile, String positivePolarityWordsFile) {
        this.negativeWordsBingLiu = readBingLiuPolarityWordList(negativePolarityWordsFile);
        this.positiveWordsBingLiu = readBingLiuPolarityWordList(positivePolarityWordsFile);
    }

    private HashSet<String> readBingLiuPolarityWordList(String polarityWordsFile) {
        HashSet<String> wordList = new HashSet<String>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(polarityWordsFile));
            String line = "";
            while ((line = br.readLine()).startsWith(";"))
                continue;
            while ((line = br.readLine()) != null) {
                wordList.add(line.trim());
            }
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

        if (wordList.size() == 0) {
            System.err.println("Attention! Didn't read any word from " + polarityWordsFile+". You won't get any sentiment information corresponding to this lexicon.");
        }
        return wordList;
    }

    private HashMap<DS_mpqaPolarityWord, String> readMpqaPolarityLexicon(String polarityFile)  {
        HashMap<DS_mpqaPolarityWord, String> hm_word_polarityMpqa = new HashMap<>();
        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(polarityFile));
            String line = "";
            while ((line = br.readLine()) != null) {
                String toks[] = line.split(" ");
                String stemmedS = toks[4].replaceFirst("stemmed1=", "");
                boolean stemmed = false;
                if (stemmedS.equals("y") || stemmedS.equals("1"))
                    stemmed = true;
                else if (stemmedS.equals("n"))
                    stemmed = false;
                else {
                    System.err.println("Could not parse " + toks[4] + " in line:" + line +" of file "+polarityFile);
                    System.err.println("Was expecting stemmed1=y or stemmed1=n");
                }

                DS_mpqaPolarityWord polWord = new DS_mpqaPolarityWord(toks[2].replaceFirst("word1=", ""),
                        toks[3].replaceFirst("pos1=", ""), stemmed);

                hm_word_polarityMpqa.put(polWord, toks[5].replaceFirst("priorpolarity=", ""));
            }
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (hm_word_polarityMpqa.keySet().size() == 0) {
            System.err.println("Attention! Didn't read any lexicon from " + polarityFile+". You won't get any sentiment information corresponding to this lexicon.");
        }
        return hm_word_polarityMpqa;
    }

    private boolean isNegated(Constituent word, View depView) {
        // determine if a word is negated using its dependency tree
        boolean isNeg = false;
        List<Constituent> c = depView.getConstituentsCovering(word);
        if (c.size() > 0) {
            List<Relation> allrel = c.get(0).getOutgoingRelations();
            if (allrel.size() > 0) {
                for (Relation rel : allrel)
                    if (rel.getRelationName().equals("neg")) {
                        isNeg = true;
                        return isNeg;
                    }
            }
        }
        return isNeg;
    }

    public int categorize(int i) {
        if(i>=1)
            return 1;
        else if(i<=-1)
            return -1;
        else
            return 0;
    }
}

