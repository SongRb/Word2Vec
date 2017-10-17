#include <iostream>



#include<string>
#include<iostream>

#include<cmath>
#include <fstream>
#include <string>
#include <functional>
#include <map>
#include <unordered_map>
#include <utility>
#include <queue>
#include <iterator>
#include <sstream>

using namespace std;

size_t getWordHash(const string& word)
{
	hash<string> str_hash;
	return str_hash(word);
}

struct BinaryTreeNode
{
	BinaryTreeNode* left = nullptr;
	BinaryTreeNode* right = nullptr;
	long long weight;
	size_t id;
	string code;
	BinaryTreeNode(long long w, size_t i, string c) :weight(w), id(i), code(std::move(c))
	{
	}
	BinaryTreeNode(BinaryTreeNode* ln, BinaryTreeNode* rn)
	{
		left = ln;
		right = rn;
		weight = ln->weight + rn->weight;
		id = static_cast<size_t>(-1);
	}
};

class BinaryTreeNodeCompare
{
public:
	bool operator()(BinaryTreeNode* lt, BinaryTreeNode* rt)
	{
		return lt->weight > rt->weight;
	}
};


struct WordInVocab
{
	long long cnt=0;
	vector<BinaryTreeNode*> point;
	string word, code="", codelen;
};




class Vocabulary
{
private:
	int minCount;
	BinaryTreeNode* root;

	string readWord(ifstream& inFile)
	{
		string word;
		inFile >> word;
		return word;
	}

	int searchVocab(string word)
	{
		if (vocab.find(getWordHash(word)) == vocab.end())
			return -1;
		return 1;
	}

	void readWordIndex(ifstream& in)
	{
		//while (!in.eof()) { string word = readWord(in);
		//cout << word << endl;
		//}
		string word;
		while (in >> word)
		{
			cout << word << endl;
		}
	}
public:
	unordered_map<size_t, WordInVocab> vocab;

	void initializeVocab()
	{
		vector<string> s = { "I","have","an","apple","an","I","I" };
		for (auto &i : s) {
			addWordToVocab(i);
		}
		cout << vocab.size() << endl;
	}


	void addWordToVocab(string word, long long increment = 1)
	{
		auto it = vocab.find(getWordHash(word));
		if (it == vocab.end())
		{
			WordInVocab tmp;
			tmp.word = word;
			tmp.cnt = increment;
			vocab.insert(make_pair(getWordHash(word), tmp));
		}
		else
		{
			it->second.cnt += increment;
		}
	}

	void reduceVocabSize()
	{
		for (auto it = vocab.begin(); it != vocab.end(); it++)
		{
			if (it->second.cnt < minCount)
			{
				vocab.erase(it);
			}
		}
	}

	void createBinaryTree()
	{
		priority_queue<BinaryTreeNode*, vector<BinaryTreeNode*>, BinaryTreeNodeCompare>pq;
		for (auto &it : vocab) {
			pq.push(new BinaryTreeNode(it.second.cnt, it.first, ""));
		}
		while (pq.size() > 1)
		{

			auto left = pq.top();
			pq.pop();
			auto right = pq.top();
			pq.pop();
			pq.push(new BinaryTreeNode(left, right));
		}


		root = pq.top();
		cout << root->weight << endl;
		vector<BinaryTreeNode*> point;
		createHuffmanCode(root,point);
	}

	void createHuffmanCode(BinaryTreeNode* n, vector<BinaryTreeNode*> point)
	{
		point.push_back(n);
		if (n->left != nullptr) { n->left->code = n->code + "0"; createHuffmanCode(n->left,point); }
		if (n->right != nullptr) { n->right->code = n->code + "1"; createHuffmanCode(n->right,point); }
		if (n->id != -1)
		{
			auto it = vocab.find(n->id);
			cout<<n->code<<endl;
			it->second.code = n->code;
			it->second.point = point;
			for (auto node : point)
			{
				cout << node <<" ";
			}
			cout << it->second.word << " " << n->code << endl;
		}
	}

	void learnFromTrainFile()
	{
		ifstream inFile;
		inFile.open("test", std::ifstream::in);
		string word;
		while (inFile >> word)
		{
			addWordToVocab(word);
		}
	}

	void saveVocab()
	{
		std::ofstream ofs;
		ofs.open("vocab", std::ofstream::out | std::ofstream::out);
		for (auto &it : vocab) {
			ofs << it.second.word << " " << it.second.cnt << endl;
		}
	}

	int getVocabSize()
	{
		return vocab.size();
	}

    unordered_map<size_t, WordInVocab>::iterator getVocabIter(string word)
    {
        return vocab.find(getWordHash(word));
    };


	void readVocab()
	{
		ifstream ifs;
		ifs.open("vocab", ifstream::in);
		string word;
		int cnt;
		for (string word, cntStr;
			getline(ifs, word, ' ') && getline(ifs, cntStr);
			)
		{
			long long cnt = stoll(cntStr);
			addWordToVocab(word, cnt);
		}
	}

//	void printStatus()
//	{
//		for each (auto it in vocab)
//		{
//			cout << it.second.word << endl;
//			for each(auto it2 in it.second.point)
//				cout << it2 << " " << endl;
//		}
//	}
    unordered_map<size_t, WordInVocab>::iterator getVocabEndIter() {
        return vocab.end();
    }
};



class Word2Vec
{
    long long layer1Size=100;
    Vocabulary vocab;
    string trainFile;
    long long trainFileSize=0;
    vector<unordered_map<size_t, WordInVocab>::iterator> table;
    const int table_size = static_cast<const int>(1e8);

    int numThreads,window,cbow,hs=0;
    long long  wordCountActual;

    vector<double> syn0, syn1, syn1neg;
    double alpha = 0.025, starting_alpha, sample=1e-3;

    long long train_words=0, iter=5;

    int negative;

    int debugMode;

    clock_t start;

    // Requires test
    void initUnigramTable()
    {
        int a, i;
        double train_words_pow = 0;
        double d1, power = 0.75;
        for (auto &it : vocab.vocab) {
            train_words_pow += pow(it.second.cnt, power);
        }

        auto it = vocab.vocab.begin();
        auto prevIt = vocab.vocab.begin();
        d1 = pow(it->second.cnt, power) / train_words_pow;

        for (a = 0; a < table_size; a++)
        {
            table[a] = it;
            if (a / (double)table_size > d1)
            {
                prevIt = it++;
                d1 += pow(it->second.cnt, power) / train_words_pow;
            }
            if (it == vocab.vocab.end()) it=prevIt;
        }
    }

    void initNet()
    {
        unsigned long long next_random = 1;
        int vocab_size = vocab.getVocabSize();
        if (hs)
        {
            syn1.resize(0);
            syn1.resize(vocab_size*layer1Size, 0);
        }
        if (negative>0)
        {
            syn1neg.resize(0);
            syn1neg.resize(vocab_size*layer1Size, 0);
        }

        for (long long a = 0; a < vocab_size; a++)
            for (long long b = 0; b < layer1Size; b++)
            {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                syn0[a * layer1Size + b] = (((next_random & 0xFFFF) / (double)65536) - 0.5) / layer1Size;
            }
        vocab.createBinaryTree();
    }

    void *trainModelThread(void *id)
    {
        vector<double> neu1(layer1Size);
        vector<double> neu1e(layer1Size);
        ifstream inFile(trainFile,ifstream::binary);
        inFile.seekg(trainFileSize / (long long)numThreads * (long long)id);
        
        long long wordCount=0,lastWordCount=0;

        vector<unordered_map<size_t, WordInVocab>::iterator> sen;
        unordered_map<size_t, WordInVocab>::iterator lastWord;
        long long sentencePosition=0;

        long long localIter = iter;

        while(true)
        {
            if (wordCount - lastWordCount > 10000)
            {
                wordCountActual += wordCount - lastWordCount;
                lastWordCount = wordCount;
                if ((debugMode > 1))
                {
                    clock_t now = clock();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                           wordCountActual / (double)(iter * train_words + 1) * 100,
                           wordCountActual / ((double)(now - start + 1) / (double)CLOCKS_PER_SEC * 1000));
                    fflush(stdout);
                }
                // Slightly decreases alpha.
                alpha = starting_alpha * (1 - wordCountActual / (double)(iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001)
                    alpha = starting_alpha * 0.0001;
            }
            // Build a sentence of word
            if(sen.empty())
            {
                string line;
                getline(inFile,line);
                std::stringstream ss(line);
                std::istream_iterator<std::string> begin(ss);
                std::istream_iterator<std::string> end;
                std::vector<std::string> senStringVec(begin, end);
                for(auto word:senStringVec)
                {
                    sen.push_back(vocab.getVocabIter(word));
                }

                wordCount+=sen.size();
                // TODO Sample words
                auto sentencePosition = 0;
            }


            if(inFile.eof()||(wordCount>train_words/numThreads))
            {
                wordCountActual+=wordCount-lastWordCount;
                localIter--;
                if(localIter==0) break;
                wordCount=0;
                lastWordCount=0;
                inFile.seekg(file_size / (long long)numThreads * (long long)id);
            }

            auto word = sen[sentencePosition];
            if(word==vocab.getVocabEndIter()) continue;

            fill(neu1.begin(),neu1.end(),0);
            fill(neu1e.begin(),neu1e.end(),0);

            int realWindowSize = rand()%window;

            if(cbow)
            {
                long long cw=0;
                for(long long i=realWindowSize; i<window*2+1-realWindowSize; i++)
                {
                    if(i!=window)
                    {
                        // TODO Check available of unmatched type for window and sentencePosition
                        long long j = sentencePosition-window+i;
                        if(j<0 || j>=sen.size()) continue;
                        lastWord = sen[j];
                        if(lastWord == vocab.getVocabEndIter()) continue;
                        for(long long k=0;k<layer1Size;k++) neu1[k]+=syn0[k+lastWord*layer1Size];
                        cw++;
                    }
                }

                if(cw)
                {
                    for(long long i=0;i<layer1Size;i++)
                    {
                        neu1[i]/=cw;
                    }

                    if(hs)
                    {
                        for(long long i=0;i<)
                                            {


                    }
                    }


                }


            }




        }









    }


    void trainModel()
    {


    }
};

int main()
{
    Vocabulary v;
    v.learnFromTrainFile();
    cout<<"Safe here"<<endl;
    v.createBinaryTree();
//    v.readVocab();
    return 0;
}




