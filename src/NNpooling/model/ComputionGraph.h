#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 256;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat;

	LinearNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_word_inputs.resize(sent_length);
	}

	inline void clear(){
		Graph::clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
		}
		_output.setParam(&model.olayer_linear);
	}
	

public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		int words_num = feature.m_tweet_words.size();
		for (int i = 0; i < words_num; i++)
		{
			_word_inputs[i].forward(this, feature.m_tweet_words[i]);
		}
		_max_pooling.forward(this, getPNodes(_word_inputs, words_num));
		_min_pooling.forward(this, getPNodes(_word_inputs, words_num));
		_avg_pooling.forward(this, getPNodes(_word_inputs, words_num));
		_concat.forward(this, &_max_pooling, &_avg_pooling, &_min_pooling);
		_output.forward(this, &_concat);
	}

};

#endif /* SRC_ComputionGraph_H_ */