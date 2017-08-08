package edu.illinois.cs.cogcomp;

import edu.illinois.cs.cogcomp.Helpers.Utils;
import edu.illinois.cs.cogcomp.Helpers.readData;
import edu.illinois.cs.cogcomp.SemLM.semLMFeatureReader;
import edu.illinois.cs.cogcomp.SentimentLM.sentimentLM_condensed;
import edu.illinois.cs.cogcomp.TopicalCoherence.TopicalCoherenceModel;
import edu.illinois.cs.cogcomp.core.utilities.configuration.Configurator;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Properties;

/**
 * Created by snigdha on 12/22/16.
 */
public class JointModelSuperClass {

    protected readData corpus = new readData();
    protected HashMap<String, double[]> hm_validSemLMFeatures= null;
    protected HashMap<String, double[]> hm_testSemLMFeatures = null;
    protected HashMap<String, double[]> hm_validSentFeatures = null;
    protected HashMap<String, double[]> hm_testSentFeatures = null;
    protected HashMap<String, double[]> hm_validTopicFeatures= null;
    protected HashMap<String, double[]> hm_testTopicFeatures = null;

    protected Properties getProperties(){
        /**************** BEGIN MODEL CONFIGURATIONS********************************************************/
        Properties p = new Properties();

        p.put("numComponents","3"); // number of semantic aspects
        p.put("extractFeaturesAgain", Configurator.TRUE);// if set to FALSE then it will read extracted features in aspect0FeaturesTrain,aspect1FeaturesTrain,aspect2FeaturesTrain,aspect0FeaturesTest,aspect1FeaturesTest,aspect2FeaturesTest

        // Input locations
        p.put("datadir","Dataset/RoCStories/");
        p.put("validationFile", p.getProperty("datadir") + "cloze_test_val__spring2016 - cloze_test_ALL_val.tsv");
        p.put("testFile", p.getProperty("datadir") + "test_spring2016.tsv");
        p.put("unannotatedFile", p.getProperty("datadir") + "100KStories.csv");

        // input for aspect 0: semLM features
        p.put("semLMFeaturesFileTest", "Resources/SemLM_Feats/test_feats.txt");
        p.put("semLMFeaturesFileVal", "Resources/SemLM_Feats/valid_feats.txt");

        // input for aspect 1: sentiment related files
        p.put("sentimentModelDump", "out/sentModel/");
        p.put("retrainSentModel", Configurator.FALSE);// do you want to retrain a sentiment language model from unannotated story corpus? Set to TRUE if out/sentModel/ is empty

        // input for aspect 2: topical Coherence related files
        p.put("gloveVectorsFile", "/shared/corpora/snigdha/preTrainedGloveVectors/glove.6B.100d.txt");
        p.put("stopWordsFile", "Resources/nltkStopwords.txt");

        p.put("aspect0FeaturesTrain", "out/semFeaturesTrain.arff");
        p.put("aspect1FeaturesTrain", "out/sentFeaturesTrain.arff");
        p.put("aspect2FeaturesTrain", "out/topicFeaturesTrain.arff");

        p.put("aspect0FeaturesTest", "out/semFeaturesTest.arff");
        p.put("aspect1FeaturesTest", "out/sentFeaturesTest.arff");
        p.put("aspect2FeaturesTest", "out/topicFeaturesTest.arff");

        // output file: stores predictions
        p.put("predictionsOutputFile", "out/predictions/predictions_HVModel.csv");
        /*************************************************************************************************/
        return p;
    }

    protected void readCorpus(Properties p) {
        // Read the corpus
        corpus.readAllData(p.getProperty("validationFile"), p.getProperty("testFile"));
        corpus.readUnannotatedData(p.getProperty("unannotatedFile"));
    }

    protected void readFeatures(Properties p){
        System.out.println("Reading semLM features");
        hm_validSemLMFeatures= readArff(p.getProperty("aspect0FeaturesTrain"));
        hm_testSemLMFeatures= readArff(p.getProperty("aspect0FeaturesTest"));

        System.out.println("Reading sentimentLM features");
        hm_validSentFeatures = readArff(p.getProperty("aspect1FeaturesTrain"));
        hm_testSentFeatures = readArff(p.getProperty("aspect1FeaturesTest"));

        System.out.println("Reading topical coherence features");
        hm_validTopicFeatures= readArff(p.getProperty("aspect2FeaturesTrain"));
        hm_testTopicFeatures = readArff(p.getProperty("aspect2FeaturesTest"));

    }

    protected void extractFeatures(Properties p) {
        // Aspect 1: Read Semantic LM features
        System.out.println("Reading semLM features");
        hm_validSemLMFeatures= semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("semLMFeaturesFileVal"));
        hm_testSemLMFeatures= semLMFeatureReader.readSemLMClassificationFeatures(p.getProperty("semLMFeaturesFileTest"));

        Utils.writeFeaturesToArffFile(corpus.valStories, hm_validSemLMFeatures, p.getProperty("aspect0FeaturesTrain"));
        Utils.writeFeaturesToArffFile(corpus.testStories, hm_testSemLMFeatures, p.getProperty("aspect0FeaturesTest"));

        // Aspect 2: extract the sentiment LM features
        System.out.println("Extracting sentimentLM features");
        System.out.println("Uncomment these lines in TwoLevelClassifiersSuperClass.java here");
        sentimentLM_condensed sentFeatExtractor = new sentimentLM_condensed();
        sentFeatExtractor.getLM(corpus.unannotatedStories, Boolean.parseBoolean(p.getProperty("retrainSentModel")), p.getProperty("sentimentModelDump"));
        hm_validSentFeatures = sentFeatExtractor.extractFeaturesForStories(corpus.valStories);
        hm_testSentFeatures = sentFeatExtractor.extractFeaturesForStories(corpus.testStories);

        Utils.writeFeaturesToArffFile(corpus.valStories, hm_validSentFeatures, p.getProperty("aspect1FeaturesTrain"));
        Utils.writeFeaturesToArffFile(corpus.testStories, hm_testSentFeatures, p.getProperty("aspect1FeaturesTest"));

        // Aspect 3: extract the topical coherence features
        System.out.println("Extracting topical coherence features");
        TopicalCoherenceModel topicalModel = new TopicalCoherenceModel(p.getProperty("stopWordsFile"), p.getProperty("gloveVectorsFile"));
        hm_validTopicFeatures = topicalModel.extractTopicFeatures(corpus.valStories);
        hm_testTopicFeatures = topicalModel.extractTopicFeatures(corpus.testStories);

        Utils.writeFeaturesToArffFile(corpus.valStories, hm_validTopicFeatures, p.getProperty("aspect2FeaturesTrain"));
        Utils.writeFeaturesToArffFile(corpus.testStories, hm_testTopicFeatures, p.getProperty("aspect2FeaturesTest"));
    }

    protected HashMap<String, double[]> readArff(String filename){
        HashMap<String, double[]> hm = new HashMap<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader(filename));
            String line = null;
            while ((line = br.readLine()) != null) {
                if (line.startsWith("@") || line.trim().length() == 0) {
                    continue;
                }
                String[] columns = line.split(",");

                // skip first column and last column is the labelMappedTo01
                int i = 1;
                double[] data = new double[columns.length - 2];
                String id = columns[0];
                if (id.startsWith("\""))
                    id = id.substring(1);
                if (id.endsWith("\""))
                    id = id.substring(0, id.length() - 1);
                for (i = 1; i < columns.length - 1; i++) {
                    data[i - 1] = Double.parseDouble(columns[i]);
                }
                hm.put(id, data);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return hm;
    }

}
