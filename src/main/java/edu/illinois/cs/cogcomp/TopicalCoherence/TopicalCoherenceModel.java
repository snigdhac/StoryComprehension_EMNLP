package edu.illinois.cs.cogcomp.TopicalCoherence;

import edu.illinois.cs.cogcomp.Helpers.Story;
import edu.illinois.cs.cogcomp.annotation.AnnotatorException;
import edu.illinois.cs.cogcomp.annotation.AnnotatorService;
import edu.illinois.cs.cogcomp.core.datastructures.ViewNames;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.Constituent;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.TextAnnotation;
import edu.illinois.cs.cogcomp.core.datastructures.textannotation.View;
import edu.illinois.cs.cogcomp.core.utilities.configuration.Configurator;
import edu.illinois.cs.cogcomp.core.utilities.configuration.ResourceManager;
import edu.illinois.cs.cogcomp.nlp.common.PipelineConfigurator;
import edu.illinois.cs.cogcomp.nlp.pipeline.IllinoisPipelineFactory;
import edu.illinois.cs.cogcomp.Helpers.readData;

import java.io.*;
import java.util.*;

/**
 * Created by snigdha on 12/12/16.
 */
public class TopicalCoherenceModel {

    private readData corpus = new readData();
    private AnnotatorService annotator = null;
    private HashMap<String, double[]> hm_gloveVectors = new HashMap<>();
    private int optionWordsNotFound=0;
    HashSet<String> stopwords = null;

    public void initializeAnnotator() {
        Properties nonDefaultProp = new Properties();
        nonDefaultProp.put(PipelineConfigurator.USE_POS.key, Configurator.TRUE);
        nonDefaultProp.put(PipelineConfigurator.USE_LEMMA.key, Configurator.TRUE);
        nonDefaultProp.put(PipelineConfigurator.USE_NER_CONLL.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_NER_ONTONOTES.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_SHALLOW_PARSE.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_STANFORD_DEP.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_STANFORD_PARSE.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_SRL_VERB.key, Configurator.FALSE);
        nonDefaultProp.put(PipelineConfigurator.USE_SRL_NOM.key, Configurator.FALSE);
        try {             // Create the AnnotatorService object
            this.annotator = IllinoisPipelineFactory.buildPipeline(new ResourceManager(nonDefaultProp));
        }catch(Exception e){
            System.err.println("There was an exception with the Annotator Service. Note annotator points to null and will create problems later with text processing.");
            e.printStackTrace();
        }
    }

    public TopicalCoherenceModel(String stopwordsFile, String gloveVectorsFile){
        initializeAnnotator();
        this.stopwords = readStopwords(stopwordsFile);
        this.readGloveVectors(gloveVectorsFile);
    }

    public HashMap<String, double[]> extractTopicFeatures(ArrayList<Story> stories) {
        ArrayList<String> posToLookFor = new ArrayList<>();
        posToLookFor.add("NN");
        posToLookFor.add("VB");
        return extractTopicFeatures(stories,posToLookFor);

    }
    public HashMap<String, double[]> extractTopicFeatures(ArrayList<Story> stories, ArrayList<String> posToLookFor) {
        HashMap<String, double[]> hm = new HashMap<>();
        int correct=0;
        for (Story story : stories) {

            List<Constituent> allTopicWords = new ArrayList<>();

            for (int i = 0; i < story.first4Sentences.length; i++) {
                String sentence = story.first4Sentences[i];

                List<Constituent> nns = getTopicWords(sentence,posToLookFor);
                allTopicWords.addAll(nns);
            }

            List<Constituent> n1 =  getTopicWords(story.option1,posToLookFor);
            List<Constituent> n2 =  getTopicWords(story.option2,posToLookFor);

            List<Constituent> s1 = n1;
            List<Constituent> s2 = n2;

            double c1 = listGloveVectorSimilarity(s1,allTopicWords,story);
            double c2 = listGloveVectorSimilarity(s2,allTopicWords,story);

            int predicted = 2;
            if(c1>c2)
                predicted = 1;
            else if(c2>c1)
                predicted = 2;
            else
                predicted = (new Random()).nextInt(1)+1;

            if(predicted==story.answer)
                correct++;

            int f3 = (c2>c1) ? 1: -1;
            hm.put(story.instanceId, new double[]{c1,c2, f3});
        }

        System.out.println("Performace of Topical Coherence Model. Correct = "+correct+" total = "+ stories.size()+" acc="+(double)correct*100/stories.size());
        System.out.println("Glove vectors not found for "+ optionWordsNotFound + " option words in all");
        return hm;
    }

    private List<String> getString(List<Constituent> s) {
        List<String> ret = new ArrayList<>();
        for(Constituent c:s)
            ret.add(c.getLabel());
        return ret;
    }

    private void readGloveVectors(String gloveVectorsFile) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(gloveVectorsFile));
            String line = null;
            while((line=br.readLine())!=null) {
                String toks[] = line.trim().split(" ");
                double vec[] = new double[toks.length-1];
                for(int i=0;i<toks.length-1;i++)
                    vec[i] = Double.parseDouble(toks[i+1]);
                hm_gloveVectors.put(toks[0].trim(),vec);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            System.out.println("read glove vectors for "+hm_gloveVectors.keySet().size()+" words.");
        }

    }

//    private static double listSimpleSimilarity(List<String> candidates, List<String> referenceList) {
//        double c1=0;
//        for(String str: candidates) {
//            if(referenceList.contains(str))
//                c1++;
//        }
//        return c1/referenceList.size();
//    }

    private double listGloveVectorSimilarity(List<Constituent> candidates, List<Constituent> referenceList, Story story) {
            List<String> stringRefList= new ArrayList<>();
            for(Constituent c:referenceList)
                stringRefList.add(c.getLabel()); // consider lemmetized form

//        System.out.println("\nProcessing "+story.instanceId);
//        System.out.println(story.pprintStory());
//        System.out.print("All topic words in context = ");
//            for(String s:stringRefList)
//                System.out.print(s+" ");
//            System.out.println();

            double avgSim=0;
            Set<Double> similarities = new HashSet<>();

            for(Constituent c: candidates) {
                if(stringRefList.contains(c.getLabel())) {
                    avgSim++;
                    similarities.add(new Double(1));
//                    System.out.println(c.getSurfaceForm()+" found exactly");
                }
                else {
                    double sim = this.maxCosSim(c, referenceList); // the gloveVetors based similarity looks at surface form of words
                    avgSim += sim; // the gloveVetors based similarity looks at surface form of words
                    similarities.add(new Double(sim));
                }
            }
            return avgSim/candidates.size();
    }

    private double minimum(Set<Double> s) {
        double min = 99999999;
        for(Double d: s)
            if(d<min)
                min=d;
        return min;
    }

    private double maxCosSim(Constituent c, List<Constituent> referenceList) {
        double[] candidateVec = hm_gloveVectors.get(c.getSurfaceForm().toLowerCase());
        double maxSim = 0;
        String maxSimStr = "";
        if(candidateVec!=null){
            for(Constituent refCons:referenceList) {
                double[] refConsVec = hm_gloveVectors.get(refCons.getSurfaceForm().toLowerCase());
                if(refConsVec!=null) {
                    double sim = this.cosSim(candidateVec, refConsVec);
                    if (sim > maxSim) {
                        maxSim = sim;
                        maxSimStr = refCons.getSurfaceForm();
                    }
                }
            }
        }
        else optionWordsNotFound++;
//        System.out.println(c.getSurfaceForm()+"=="+maxSimStr+" "+maxSim+" ");//////////
        return maxSim;
    }

    private double cosSim(double[] v1, double[] v2) {
        double sum = 0, s1=0, s2=0;
        for(int i=0;i<v1.length;i++){
            sum += v1[i]*v2[i];
            s1 += v1[i]*v1[i];
            s2 += v2[i]*v2[i];
        }
        return sum/(Math.sqrt(s1)*Math.sqrt(s2));
    }

    private String printList(List<Constituent> l){
        String ret = "";
        for(Constituent s:l)
            ret+= s.getLabel()+" ";
        return ret;
    }

    private  List<Constituent> getTopicWords(String sentence, ArrayList<String> posToLookFor) {
        List<Constituent> ret = new ArrayList<>();
        try {
            TextAnnotation ta = annotator.createBasicTextAnnotation("sample", "id", sentence);
            annotator.addView(ta, ViewNames.POS);
            annotator.addView(ta, ViewNames.LEMMA);
            View posView = ta.getView(ViewNames.POS);
            View lemmView = ta.getView(ViewNames.LEMMA);

            for (Constituent wordLemm : lemmView) {
                String wordSurfaceForm = wordLemm.getSurfaceForm();
                String posCode = posView.getConstituentsCovering(wordLemm).get(0).getLabel();

                for(String pos: posToLookFor)
                    if(posCode.startsWith(pos)) {
//                    if(!this.stopwords.contains(wordLemm.getSurfaceForm().toLowerCase()))
                        ret.add(wordLemm);
                        break;
                    }

            }
        } catch (AnnotatorException e){
                e.printStackTrace();
            }

        return ret;
        }

    public static HashSet<String> readStopwords(String stopwordsFile) {
        HashSet<String> ret = new HashSet();
        try {
            BufferedReader br = new BufferedReader(new FileReader(stopwordsFile));
            String line = null;
            while((line=br.readLine())!=null){
                ret.add(line.toLowerCase());
            }
            br.close();
        } catch (FileNotFoundException e) {
            System.err.println("Could not find "+stopwordsFile);
            System.err.println("Code will proceed without using stopwords.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }
    }
