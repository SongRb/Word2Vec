#include <iostream>

#include<cmath>
#include <fstream>
#include <functional>
#include <map>
#include <unordered_map>
#include <queue>
#include <iterator>
#include <sstream>
#include <thread>
#include <random>
#include <algorithm>
#include <cassert>

using namespace std;

size_t getWordHash(const string &word)
{
    hash<string> str_hash;
    return str_hash(word);
}

struct BinaryTreeNode
{
    BinaryTreeNode *left = nullptr;
    BinaryTreeNode *right = nullptr;
    long long weight;
    long long id;
    string code;

    BinaryTreeNode(long long w, long long i, string c) : weight(w), id(i), code(std::move(c))
    {
    }

    BinaryTreeNode(BinaryTreeNode *ln, BinaryTreeNode *rn)
    {
        left = ln;
        right = rn;
        weight = ln->weight + rn->weight;
        id = static_cast<long long>(-1);
    }
};


class Vocabulary
{
private:
    int minCount = 1;
    BinaryTreeNode *root;
    vector<BinaryTreeNode *> midNodeList;
    string trainFileName;
public:
    struct Word
    {
        long long cnt = 0;
        BinaryTreeNode *currentNode;
        // point is to identify vector
        vector<BinaryTreeNode *> point;
        string word, code = "";
    };

    explicit Vocabulary(string fileName) : trainFileName(std::move(fileName)), root(nullptr)
    {
    }

    string readWord(ifstream &inFile)
    {
        string word;
        inFile >> word;
        return word;
    }

    int searchVocab(const string &word)
    {
        if (wordTable.find(getWordHash(word)) == wordTable.end())
        {
            return -1;
        }
        return 1;
    }

    void readWordIndex(ifstream &in)
    {
        string word;
        while (in >> word)
        {
            cout << word << endl;
        }
    }

public:
    unordered_map<size_t, Word> wordTable;  ///< word hash table

    void initializeVocab()
    {
        vector<string> s = {"I", "have", "an", "apple", "an", "I", "I"};
        for (auto &i : s)
        {
            addWordToVocab(i);
        }
        cout << wordTable.size() << endl;
    }

    void addWordToVocab(const string &word, long long increment = 1)
    {
        auto it = wordTable.find(getWordHash(word));
        if (it == wordTable.end())
        {
            Word tmp;
            tmp.word = word;
            tmp.cnt = increment;
            wordTable.insert(make_pair(getWordHash(word), tmp));
        } else
        {
            it->second.cnt += increment;
        }
    }

    void reduceVocabSize()
    {
        for (auto it = wordTable.begin(); it != wordTable.end(); it++)
        {
            if (it->second.cnt < minCount)
            {
                wordTable.erase(it);
            }
        }
    }

    void createBinaryTree()
    {
        auto comp = []( BinaryTreeNode * lhs, BinaryTreeNode * rhs ) { return lhs->weight > rhs->weight; };
        priority_queue<BinaryTreeNode*, vector<BinaryTreeNode *>, decltype(comp)> pq(comp);
        for (auto &it : wordTable)
        {
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
        vector<BinaryTreeNode *> point;
        createHuffmanCode(root, point);
        long long length = static_cast<long long>(midNodeList.size());
        for(long long i=0;i<length;i++)
        {
            midNodeList[i]->id = static_cast<long long>(i);
        }

    }

    void createHuffmanCode(BinaryTreeNode *n, vector<BinaryTreeNode *> point)
    {
        point.push_back(n);

        if (n->left != nullptr)
        {
            n->left->code = n->code + "0";
            createHuffmanCode(n->left, point);
        }
        if (n->right != nullptr)
        {
            n->right->code = n->code + "1";
            createHuffmanCode(n->right, point);
        }
        if (n->id != -1)
        {
            auto it = wordTable.find(n->id);
            it->second.code = n->code;
            it->second.point = point;
            it->second.currentNode = n;
        } else
        {
            midNodeList.push_back(n);
        }
    }

    void learnFromTrainFile()
    {
        ifstream inFile;
        inFile.open(trainFileName, std::ifstream::in);
        string word;
        while (inFile >> word)
        {
            addWordToVocab(word);
        }
    }

    void saveVocab()
    {
        std::ofstream ofs;
        ofs.open("wordTable", std::ofstream::out);
        for (auto &it : wordTable)
        {
            ofs << it.second.word << " " << it.second.cnt << endl;
        }
    }

    int getVocabSize()
    {
        return static_cast<int>(wordTable.size());
    }

    /// Wrapper for get an iterator to given word in `Vocabulary`
    /// \param word
    /// \return an `iterator` to word
    unordered_map<size_t, Word>::iterator getVocabIter(const string &word)
    {
        return wordTable.find(getWordHash(word));
    };

    void readVocab()
    {
        ifstream ifs;
        ifs.open("wordTable", ifstream::in);
        for (string word, cntStr;
             getline(ifs, word, ' ') && getline(ifs, cntStr);
            )
        {
            long long cnt = stoll(cntStr);
            addWordToVocab(word, cnt);
        }
    }

    void printMidNodeList()
    {
        for (auto it : midNodeList)
        {
            cout << it << endl;

        }
    }

    unordered_map<size_t, Word>::iterator endIter()
    {
        return wordTable.end();
    }

    unordered_map<long long, vector<double>> getMidNodeMatrix(long long size)
    {
        unordered_map<long long, vector<double>> mat;
        for (auto node:midNodeList)
        {
            mat.insert(make_pair((long long) node, vector<double>(static_cast<unsigned long>(size), 0)));
        }
        return mat;
    };

};

class Word2Vec
{
public:
    long long layer1Size = 100;
    Vocabulary vocab;
    string trainFile;
    long long trainFileSize = 0;
    vector<unordered_map<size_t, Vocabulary::Word>::iterator> table;
    const int table_size = static_cast<const int>(1e8);

    int numThreads = 2, window = 5, cbow = 1, hs = 0;
    long long wordCountActual = 0;

    vector<double> expTable;

    const int EXP_TABLE_SIZE = 1000;
    const int MAX_EXP = 6;

    unordered_map<long long, vector<double>> syn0; ///< Original vector table index by word huffman code
    unordered_map<long long, vector<double>> syn1; ///< Parameter from hidden state to node
    unordered_map<long long, vector<double>> syn1neg;

    double alpha = 0.025, starting_alpha, sample = 1e-3;

    int classes = 0;

    long long totalWordCount = 0, iterTimes = 100;
    /// Negative sampling
    int ns = 5;

    int debugMode = 2;

    clock_t start;
public:
    explicit Word2Vec(const string &trainFileName) : trainFile(trainFileName), vocab(trainFileName)
    {
        vocab.learnFromTrainFile();
        vocab.createBinaryTree();
        ns = 1;
        for (int i = 0; i < EXP_TABLE_SIZE; i++)
        {
            expTable.push_back(exp((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)); // Precompute the exp() table
            expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
        }
        initNet();
    }

    // Requires test
    void initUnigramTable()
    {
        int a, i;
        double train_words_pow = 0;
        double d1, power = 0.75;
        for (auto &it : vocab.wordTable)
        {
            train_words_pow += pow(it.second.cnt, power);
        }

        auto it = vocab.wordTable.begin();
        auto prevIt = vocab.wordTable.begin();
        d1 = pow(it->second.cnt, power) / train_words_pow;

        for (a = 0; a < table_size; a++)
        {
            table.push_back(it);
            if (a / (double) table_size > d1)
            {
                prevIt = it++;
                d1 += pow(it->second.cnt, power) / train_words_pow;
            }
            if (it == vocab.wordTable.end())
            { it = prevIt; }
        }
    }

public:
    void initNet()
    {
        default_random_engine defaultRandomEngine;
        int vocab_size = vocab.getVocabSize();
        if (hs)
        {
            syn1 = vocab.getMidNodeMatrix(layer1Size);
        }
        uniform_real_distribution<double> dist(-0.5, 0.5);
        for (auto n:vocab.wordTable)
        {
            vector<double> tmp(static_cast<unsigned long>(layer1Size));
            std::generate(tmp.begin(), tmp.end(), std::bind(dist, std::ref(defaultRandomEngine)));
            syn0.insert(make_pair((long long) n.second.currentNode, tmp));
        }
        if (ns > 0)
        {
            for (auto n:vocab.wordTable)
            {
                vector<double> tmp(static_cast<unsigned long>(layer1Size));
                std::generate(tmp.begin(), tmp.end(), std::bind(dist, std::ref(defaultRandomEngine)));
                syn1neg.insert(make_pair((long long) n.second.currentNode, tmp));
            }
        }
    }

    const unordered_map<long long, vector<double>>::iterator getVector(const string &word)
    {
        cout << "Find vector representation of " << word << endl;
        auto iter = vocab.wordTable.find(getWordHash(word));
        cout << iter->second.word << endl;
        return syn0.find((long long) iter->second.currentNode);
    }

    int getVocabSize()
    { return vocab.getVocabSize(); }

    unordered_map<size_t, Vocabulary::Word>::iterator getVocabIter(const string &word)
    {
        return vocab.wordTable.find(getWordHash(word));
    }

    void decreaseAlpha()
    {
        alpha =
            starting_alpha * (1 - wordCountActual / (double) (iterTimes * totalWordCount + 1));
        if (alpha < starting_alpha * 0.0001)
        {
            alpha = starting_alpha * 0.0001;
        }
    }

    void printProgress()
    {
        clock_t now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               wordCountActual / (double) (iterTimes * totalWordCount + 1) * 100,
               wordCountActual / ((double) (now - start + 1) / (double) CLOCKS_PER_SEC * 1000));
        fflush(stdout);
    }

};

void verifyResult(Word2Vec &w2v);
void buildSentence(Word2Vec &w2v,
                   vector<unordered_map<size_t, Vocabulary::Word>::iterator> &sentence,
                   ifstream &inFile);
void negativeSampling(Word2Vec &w2v,
                      const vector<double> &neu1,
                      const vector<unordered_map<size_t, Vocabulary::Word>::iterator> &sentence,
                      unordered_map<size_t, Vocabulary::Word>::iterator &lastWord,
                      long long int wordIndex,
                      unordered_map<size_t, Vocabulary::Word>::iterator &word,
                      int realWindowSize,
                      vector<double> &neu1e,
                      default_random_engine &defaultRandomEngine);
void hierarchicalSoftmax(Word2Vec &w2v, const vector<double> &neu1, vector<double> &neu1e,
                         const unordered_map<size_t, Vocabulary::Word>::iterator &word);
void trainModelThread(Word2Vec &w2v, int threadId)
{
    /// Average of vector in window
    vector<double> neu1(static_cast<unsigned long>(w2v.layer1Size));
    vector<double> neu1e(static_cast<unsigned long>(w2v.layer1Size));
    ifstream inFile(w2v.trainFile, ifstream::binary);
    inFile.seekg(w2v.trainFileSize / (long long) w2v.numThreads * (long long) threadId);
    default_random_engine defaultRandomEngine;

    long long wordCount = 0, lastWordCount = 0;
    /// `vector` of iterator of word, representing a sentence
    vector<unordered_map<size_t, Vocabulary::Word>::iterator> sentence;
    unordered_map<size_t, Vocabulary::Word>::iterator lastWord;
    /// Index of central word
    long long wordIndex = 0;

    long long localIterTimes = w2v.iterTimes;

    while (true)
    {
        if (w2v.debugMode > 1)
        {
            cout << "wordCount: " << wordCount << endl;
            cout << "lastWordCount: " << lastWordCount << endl;
        }

        if (wordCount - lastWordCount > 10000)
        {
            w2v.wordCountActual += wordCount - lastWordCount;
            lastWordCount = wordCount;
            if (w2v.debugMode > 0)
            {
                w2v.printProgress();
            }
            // Slightly decreases alpha.
            w2v.decreaseAlpha();
        }
        // Build a sentence
        if (sentence.empty())
        {
            buildSentence(w2v, sentence, inFile);
            wordCount += sentence.size();
            wordIndex = 0;
        }

        if (inFile.eof() || (wordCount > w2v.totalWordCount / w2v.numThreads))
        {
            w2v.wordCountActual += wordCount - lastWordCount;
            localIterTimes--;
            if (w2v.debugMode > 1)
            { cout << "Local Iterator: " << localIterTimes << endl; }

            if (localIterTimes == 0)
            { break; }
            wordCount = 0;
            lastWordCount = 0;
            inFile.seekg(w2v.trainFileSize / (long long) w2v.numThreads * (long long) threadId);
        }

        auto word = sentence[wordIndex];
        if (word == w2v.vocab.endIter())
        { continue; }

        fill(neu1.begin(), neu1.end(), 0);
        fill(neu1e.begin(), neu1e.end(), 0);
        uniform_int_distribution<> dist(1, w2v.window - 1);
        int realWindowSize = dist(defaultRandomEngine);

        if (w2v.cbow)
        {
            /// Count of word in window
            long long localWordCount = 0;
            // Calculate the window of central word
            for (long long i = realWindowSize; i < w2v.window * 2 + 1 - realWindowSize; i++)
            {
                if (i != w2v.window)
                {
                    // TODO Check available of unmatched type for window and wordIndex
                    long long j = wordIndex - w2v.window + i;
                    if (j < 0 || j >= sentence.size())
                    { continue; }
                    lastWord = sentence[j];
                    if (lastWord == w2v.vocab.endIter())
                    { continue; }
                    /// Index of word in vector table
                    auto index = (long long) lastWord->second.currentNode;

                    // Calculate sum of word vector in window
                    for (long long k = 0; k < w2v.layer1Size; k++)
                    { neu1[k] += w2v.syn0[index][k]; }
                    localWordCount++;
                }
            }

            if (localWordCount)
            {
                for (long long i = 0; i < w2v.layer1Size; i++)
                {
                    neu1[i] /= localWordCount;
                }
                if (w2v.hs)
                {
                    hierarchicalSoftmax(w2v, neu1, neu1e, word);
                }

                if (w2v.ns > 0)
                {
                    negativeSampling(w2v, neu1, sentence, lastWord, wordIndex, word, realWindowSize, neu1e,
                                     defaultRandomEngine);
                }
            }
        }
        wordIndex++;
        // End of processing the sentence
        if (wordIndex >= sentence.size())
        {
            wordIndex = 0;
            sentence.erase(sentence.begin(), sentence.end());
            continue;
        }
    }
}
///      如果采用分层softmax优化
//       根据Huffman树上从根节点到当前词的叶节点的路径，遍历所有经过的中间节点
void hierarchicalSoftmax(Word2Vec &w2v, const vector<double> &neu1, vector<double> &neu1e,
                         const unordered_map<size_t, Vocabulary::Word>::iterator &word)
{
    for (size_t i = 0; i < word->second.code.size(); i++)
    {
        /// Probability of each node
        double f = 0;
        auto index = (long long) word->second.point[i];
        for (int j = 0; j < w2v.layer1Size; j++)
        { f += w2v.syn1[index][j] * neu1[j]; }
        if (f <= -w2v.MAX_EXP or f >= w2v.MAX_EXP)
        { continue; }
        else
        { f = w2v.expTable[(int) ((f + w2v.MAX_EXP) * (w2v.EXP_TABLE_SIZE / w2v.MAX_EXP / 2))]; }

        double g = (1 - word->second.code[i] - f) * w2v.alpha; ///< gradient*learning rate
        // Propagate errors output -> hidden
        for (int j = 0; j < w2v.layer1Size; j++)
        { neu1e[j] += g * w2v.syn1[index][j]; }
        // Learn weights hidden -> output
        for (int j = 0; j < w2v.layer1Size; j++)
        { w2v.syn1[index][j] += g * neu1[j]; }
    }
}
void negativeSampling(Word2Vec &w2v,
                      const vector<double> &neu1,
                      const vector<unordered_map<size_t, Vocabulary::Word>::iterator> &sentence,
                      unordered_map<size_t, Vocabulary::Word>::iterator &lastWord,
                      long long int wordIndex,
                      unordered_map<size_t, Vocabulary::Word>::iterator &word,
                      int realWindowSize,
                      vector<double> &neu1e,
                      default_random_engine &defaultRandomEngine)
{
    uniform_int_distribution<unsigned long> dist2(0, w2v.table.size() - 1);
    for (int i = 0; i < w2v.ns + 1; i++)
    {
        unordered_map<size_t, Vocabulary::Word>::iterator target;
        long long label;
        // Positive sample
        if (i == 0)
        {
            target = word;
            label = 1;
        }
            // Negative sample
        else
        {
            target = w2v.table[dist2(defaultRandomEngine)];
            if (target == word)
            { continue; }
            label = 0;
        }

        auto index = (long long) target->second.currentNode;
        double f = 0;
        for (int j = 0; j < w2v.layer1Size; j++)
        { f += neu1[j] * w2v.syn1neg[index][j]; }
        double g;

        if (f > w2v.MAX_EXP)
        { g = (label - 1) * w2v.alpha; }
        else if (f < -w2v.MAX_EXP)
        { g = (label - 0) * w2v.alpha; }
        else
        {
            g = (label -
                w2v.expTable[(int) ((f + w2v.MAX_EXP) * (w2v.EXP_TABLE_SIZE / w2v.MAX_EXP / 2))]) *
                w2v.alpha;
        }
        // Propagate errors output -> hidden
        for (int j = 0; j < w2v.layer1Size; j++)
        { neu1e[j] += g * w2v.syn1neg[index][j]; }
        // Learn weights hidden -> output
        for (int j = 0; j < w2v.layer1Size; j++)
        { w2v.syn1neg[index][j] += g * neu1[j]; }

    }
    // BP from hidden layer to word2vec layer
    for (long long a = realWindowSize; a < w2v.window * 2 + 1 - realWindowSize; a++)
    {
        if (a != w2v.window)
        {
            long long c = wordIndex - w2v.window + a;
            if (c < 0)
            { continue; }
            if (c >= sentence.size())
            { continue; }
            lastWord = sentence[c];
            if (lastWord == w2v.vocab.endIter())
            { continue; }

            auto l = (long long) lastWord->second.currentNode;

            for (c = 0; c < w2v.layer1Size; c++)
            { w2v.syn0[l][c] += neu1e[c]; }
        }
    }
}
void buildSentence(Word2Vec &w2v,
                   vector<unordered_map<size_t, Vocabulary::Word>::iterator> &sentence,
                   ifstream &inFile)
{
    string line;
    getline(inFile, line);
    stringstream ss(line);
    istream_iterator<string> begin(ss);
    istream_iterator<string> end;
    vector<string> senStringVec(begin, end);
    for (const auto &word:senStringVec)
    {
        // TODO Sample words
        sentence.push_back(w2v.getVocabIter(word));
    }
}

vector<double> getDifference(const vector<double> &v1, const vector<double> &v2)
{
    assert(v1.size() == v2.size());
    vector<double> c;
    auto v1Iter = v1.begin();
    auto v2Iter = v2.begin();
    while (v1Iter != v1.end())
    {
        c.push_back(*v1Iter - *v2Iter);
        v1Iter++;
        v2Iter++;
    }
    return c;

}

double getDotProduct(const vector<double> &v1, const vector<double> &v2)
{
    assert(v1.size() == v2.size());
    double c = 0;
    double vc1 = 0, vc2 = 0;

    auto v1Iter = v1.begin();
    auto v2Iter = v2.begin();
    while (v1Iter != v1.end())
    {
        c += (*v1Iter * *v2Iter);
        vc1 += (*v1Iter * *v1Iter);
        vc2 += (*v2Iter * *v2Iter);
        v1Iter++;
        v2Iter++;
    }
    return c / (sqrt(vc1) * sqrt(vc2));
}

void printVector(const vector<double> &v)
{
    for (auto it:v)
    { cout << it << " "; }
    cout << endl;
}

void trainModel(Word2Vec &w2v)
{
    w2v.starting_alpha = w2v.alpha;

    w2v.initNet();

    if (w2v.ns > 0)
    { w2v.initUnigramTable(); }
    w2v.start = clock();

    vector<thread> threadPool;

    for (int i = 0; i < w2v.numThreads; i++)
    {
        threadPool.emplace_back(trainModelThread, ref(w2v), i);
    }

    for (int i = 0; i < w2v.numThreads; i++)
    {
        threadPool[i].join();
    }

    cout << "Train finished" << endl;

    ofstream outFile;
    outFile.open("result", std::ofstream::out);
    if (w2v.classes == 0)
    {
        outFile << w2v.getVocabSize() << "\t" << w2v.layer1Size << endl;
        for (auto it:w2v.vocab.wordTable)
        {
            outFile << it.second.word << "\t";
            auto index = (long long) it.second.currentNode;
            for (int j = 0; j < w2v.layer1Size; j++)
            {
                outFile << w2v.syn0[index][j] << " ";
            }
            outFile << endl;
        }
        outFile << endl;
    }
}
void verifyResult(Word2Vec &w2v)
{
    vector<string> stringList = {"man", "king", "woman", "queen"};
    auto iter = w2v.getVector(stringList[0]);

    auto diff1 = getDifference(w2v.getVector(stringList[0])->second, w2v.getVector(stringList[1])->second);
    auto diff2 = getDifference(w2v.getVector(stringList[2])->second, w2v.getVector(stringList[3])->second);
    cout << "Product: " << getDotProduct(diff1, diff2) << endl;
}

void findSimiliarWord(Word2Vec &w2v, string word)
{
    for(const auto& candidate:w2v.syn0)
    {

    }
}



int main()
{
    Word2Vec word2Vec("test");
    word2Vec.debugMode=0;
    word2Vec.numThreads = 4;
    trainModel(word2Vec);
    verifyResult(word2Vec);
//    const int EXP_TABLE_SIZE=1000;
//    vector<double>expTable;
//    const int MAX_EXP = 6;



//    v.readVocab();
    return 0;
}




