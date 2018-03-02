# StoryComprehension_EMNLP

Quick start-up:
1. Get this project on your machine
2. Download glove.6B.100d.txt from https://nlp.stanford.edu/projects/glove/ and place it in Resources/preTrainedGloveVectors.
3. Inside this extracted folder, do:

    mvn clean
    
    mvn install
    
    mvn -e exec:java -Dexec.mainClass=edu.illinois.cs.cogcomp.Run4_ContextAwareHVModel

This runs the proposed model. See details below to troubleshoot and try different settings and models.

*****************************************************
This code was used for the following paper:

Snigdha Chaturvedi, Haoruo Peng, and Dan Roth, 'Story Comprehension for Predicting What Happens Next', EMNLP 2017

Before running the code, ensure that:

a. the data is stored in Dataset/RoCStories/

b. SemLM features are extracted and stored in Resources/SemLM_Feats/

c. out/sentModel/ is populated. Otherwise change properties (see below) to "p.put("retrainSentModel", Configurator.TRUE);" to retrain a sentiment language model using the given corpus of unannotated stories. This might take a while.

d. out/predictions folder exists

e. download the file glove.6B.100d.txt and place it in Resources/preTrainedGloveVectors. Check ReadMe.txt in that folder to know where to get it from.

*******************************************************

For obtaining results for various models, the relevant main functions are:

1. For LR (baseline) -- edu.illinois.cs.cogcomp.FlatClassifier

Output: out/predictions/predictions_LR.csv // predictions of LR model



2. For Aspect Aware Ensemble (baseline) -- edu.illinois.cs.cogcomp.DisJointModel

Output: out/predictions/predictions_AspectAwareSoftEnsemble.csv // predictions of this model



3. For Majority Voting and Soft Voting (baselines) -- edu.illinois.cs.cogcomp.RunVotingModel

Output: out/predictions/predictions_HardVoting.csv  // Predictions of Majority Voting

	out/predictions/predictions_SoftVoting.csv // Predictions of Soft Voting
		

4. For Hidden Coherence Model (proposed approach)-- edu.illinois.cs.cogcomp.Run4_ContextAwareHVModel

Output: out/predictions/predictions_HVModel.csv // predictions of this model

Note: You can change the parameters of this model in edu.illinois.cs.cogcomp.JointModelSuperClass

All models (expect LR) also create the following files (containing features) which can be reused for other models: 

out/semFeaturesTrain.arff, out/sentFeaturesTrain.arff, out/topicFeaturesTrain.arff, out/semFeaturesTest.arff, out/sentFeaturesTest.arff, out/topicFeaturesTest.arff

Each main() has a few lines which set the properties/configurations.

********************************************************

If you want to use this code for another dataset, you will have to 

(i) specify the "datadir" (change the properties) which should contain the  "validationFile" (training stories), "testFile" (test stories), and "unannotatedFile" (unannotated corpus which is used for learning sentiment LM). Note that the code assumes a specific format for the stories, and reads data in that format using readAnnotatedFile() and readUnannotatedData() in edu.illinois.cs.cogcomp.Helpers.readData. 

(ii) The code assumes that you have the features extracted from the SemLM model and stored in Resources/SemLM_Feats/valid_feats.txt and Resources/SemLM_Feats/test_feats.txt. Each line represents a story and contains the story id followed by triples feature_o1, feature_o2, comparitiveFeature for each feature type. Hence if there are N feature, you have 3N columns (excluding story id). Here, comparative feature is 1 if feature_o2>feature_o1, and -1 otherwise. The SemLM modell is described in Haoruo Peng, Dan Roth, 'Two Discourse Driven Language Models for Semantics', ACL 2017. 



