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
    size_t id;
    string code;

    BinaryTreeNode(long long w, size_t i, string c) : weight(w), id(i), code(std::move(c))
    {
    }

    BinaryTreeNode(BinaryTreeNode *ln, BinaryTreeNode *rn)
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
    bool operator()(BinaryTreeNode *lt, BinaryTreeNode *rt)
    {
        return lt->weight > rt->weight;
    }
};


struct WordInVocab
{
    long long cnt = 0;
    BinaryTreeNode *currentNode;
    // point is to identify vector
    vector<BinaryTreeNode *> point;
    string word, code = "";
};


class Vocabulary
{
private:
    int minCount = 1;
    BinaryTreeNode *root;
    vector<BinaryTreeNode *> midNodeList;
    string trainFileName;
public:
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
        if (vocab.find(getWordHash(word)) == vocab.end())
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
    unordered_map<size_t, WordInVocab> vocab;

    void initializeVocab()
    {
        vector<string> s = {"I", "have", "an", "apple", "an", "I", "I"};
        for (auto &i : s)
        {
            addWordToVocab(i);
        }
        cout << vocab.size() << endl;
    }


    void addWordToVocab(const string &word, long long increment = 1)
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
        priority_queue<BinaryTreeNode *, vector<BinaryTreeNode *>, BinaryTreeNodeCompare> pq;
        for (auto &it : vocab)
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
            auto it = vocab.find(n->id);
            cout << n->code << endl;
            it->second.code = n->code;
            it->second.point = point;
            it->second.currentNode = n;
            for (auto node : point)
            {
                cout << node << " ";
            }
            cout << it->second.word << " " << n->code << endl;
        }
        else
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
        ofs.open("vocab", std::ofstream::out);
        for (auto &it : vocab)
        {
            ofs << it.second.word << " " << it.second.cnt << endl;
        }
    }

    int getVocabSize()
    {
        return static_cast<int>(vocab.size());
    }

    unordered_map<size_t, WordInVocab>::iterator getVocabIter(const string &word)
    {
        return vocab.find(getWordHash(word));
    };


    void readVocab()
    {
        ifstream ifs;
        ifs.open("vocab", ifstream::in);
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

    unordered_map<size_t, WordInVocab>::iterator getVocabEndIter()
    {
        return vocab.end();
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
    vector<unordered_map<size_t, WordInVocab>::iterator> table;
    const int table_size = static_cast<const int>(1e8);

    int numThreads = 1, window = 5, cbow = 1, hs = 0;
    long long wordCountActual = 0;

    vector<double> expTable;

    const int EXP_TABLE_SIZE = 1000;
    const int MAX_EXP = 6;

    unordered_map<long long, vector<double>> syn0New, syn1New, syn1negNew;


    double alpha = 0.025, starting_alpha, sample = 1e-3;

    int classes = 0;

    long long train_words = 0, iter = 5, label;

    int negative = 5;

    int debugMode = 2;

    clock_t start;
public:
    explicit Word2Vec(const string &trainFileName) : trainFile(trainFileName), vocab(trainFileName)
    {
        vocab.learnFromTrainFile();
        vocab.createBinaryTree();
        negative = 1;
        for (int i = 0; i < EXP_TABLE_SIZE; i++)
        {
            expTable.push_back(0);
            expTable[i] = exp((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
            expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
        }
    }

    // Requires test
    void initUnigramTable()
    {
        int a, i;
        double train_words_pow = 0;
        double d1, power = 0.75;
        for (auto &it : vocab.vocab)
        {
            train_words_pow += pow(it.second.cnt, power);
        }

        auto it = vocab.vocab.begin();
        auto prevIt = vocab.vocab.begin();
        d1 = pow(it->second.cnt, power) / train_words_pow;

        for (a = 0; a < table_size; a++)
        {
            table.push_back(it);
            if (a / (double) table_size > d1)
            {
                prevIt = it++;
                d1 += pow(it->second.cnt, power) / train_words_pow;
            }
            if (it == vocab.vocab.end())
            { it = prevIt; }
        }
    }

public:
    void initNet()
    {
        unsigned long long next_random = 1;
        int vocab_size = vocab.getVocabSize();
        if (hs)
        {
//            syn1.resize(0);
//            syn1.resize(static_cast<unsigned long>(vocab_size * layer1Size), 0);
            syn1New = vocab.getMidNodeMatrix(layer1Size);
        }
        for (auto n:vocab.vocab)
        {
            vector<double> tmp;
            random_device rd;
            mt19937 mt(rd());
            uniform_real_distribution<double> dist(-0.5, 0.5);
            for (long long b = 0; b < layer1Size; b++)
            {

                tmp.push_back(dist(mt));
                cout << "Random number " << b << " : " << tmp.back() << endl;
            }
            syn0New.insert(make_pair((long long) n.second.currentNode, tmp));
        }
        if (negative > 0)
        {
//            syn1neg.resize(0);
//            syn1neg.resize(static_cast<unsigned long>(vocab_size * layer1Size), 0);
            syn1negNew = syn0New;
        }

//        for (long long a = 0; a < vocab_size; a++)
//            for (long long b = 0; b < layer1Size; b++)
//            {
//                next_random = next_random * (unsigned long long) 25214903917 + 11;
//                syn0[a * layer1Size + b] = (((next_random & 0xFFFF) / (double) 65536) - 0.5) / layer1Size;
//            }


        cout << syn1negNew.size() << endl;
    }

    unordered_map<long long, vector<double>>::iterator getVector(const string &word)
    {
        cout << "Find vector representation of " << word << endl;
        auto iter = vocab.vocab.find(getWordHash(word));
        cout << iter->second.word << endl;
        return syn0New.find((long long) iter->second.currentNode);
    }
};

void trainModelThread(Word2Vec &w2v, int id)
{
    vector<double> neu1(static_cast<unsigned long>(w2v.layer1Size));
    vector<double> neu1e(static_cast<unsigned long>(w2v.layer1Size));
    ifstream inFile(w2v.trainFile, ifstream::binary);
    inFile.seekg(w2v.trainFileSize / (long long) w2v.numThreads * (long long) id);

    long long wordCount = 0, lastWordCount = 0;

    vector<unordered_map<size_t, WordInVocab>::iterator> sentence;
    unordered_map<size_t, WordInVocab>::iterator lastWord;
    long long l1, l2;
    long long sentencePosition = 0;

    long long localIter = w2v.iter;

    while (true)
    {
        cout << "wordCount: " << wordCount << endl;
        cout << "lastWordCount: " << lastWordCount << endl;
        if (wordCount - lastWordCount > 10000)
        {
            w2v.wordCountActual += wordCount - lastWordCount;
            lastWordCount = wordCount;
            if ((w2v.debugMode > 1))
            {
                clock_t now = clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, w2v.alpha,
                       w2v.wordCountActual / (double) (w2v.iter * w2v.train_words + 1) * 100,
                       w2v.wordCountActual / ((double) (now - w2v.start + 1) / (double) CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            // Slightly decreases alpha.
            w2v.alpha = w2v.starting_alpha * (1 - w2v.wordCountActual / (double) (w2v.iter * w2v.train_words + 1));
            if (w2v.alpha < w2v.starting_alpha * 0.0001)
            {
                w2v.alpha = w2v.starting_alpha * 0.0001;
            }
        }
        // Build a sentence of word
        if (sentence.empty())
        {
            string line;
            getline(inFile, line);
            std::stringstream ss(line);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> senStringVec(begin, end);
            for (const auto &word:senStringVec)
            {
                sentence.push_back(w2v.vocab.getVocabIter(word));
            }

            wordCount += sentence.size();
            // TODO Sample words
            sentencePosition = 0;
        }


        if (inFile.eof() || (wordCount > w2v.train_words / w2v.numThreads))
        {
            w2v.wordCountActual += wordCount - lastWordCount;
            localIter--;
            cout << "Local Iterator: " << localIter << endl;
            if (localIter == 0)
            { break; }
            wordCount = 0;
            lastWordCount = 0;
            inFile.seekg(w2v.trainFileSize / (long long) w2v.numThreads * (long long) id);
        }

        auto word = sentence[sentencePosition];
        if (word == w2v.vocab.getVocabEndIter())
        { continue; }

        fill(neu1.begin(), neu1.end(), 0);
        fill(neu1e.begin(), neu1e.end(), 0);
        random_device rd;
        mt19937 mt(rd());
        uniform_int_distribution<int> dist(1, w2v.window - 1);
        int realWindowSize = dist(mt);

        if (w2v.cbow)
        {
            long long cw = 0;
            // Calculate the window of central word
            for (long long i = realWindowSize; i < w2v.window * 2 + 1 - realWindowSize; i++)
            {
                if (i != w2v.window)
                {
                    // TODO Check available of unmatched type for window and sentencePosition
                    long long j = sentencePosition - w2v.window + i;
                    if (j < 0 || j >= sentence.size())
                    { continue; }
                    lastWord = sentence[j];
                    if (lastWord == w2v.vocab.getVocabEndIter())
                    { continue; }
                    l1 = (long long) lastWord->second.currentNode;

                    // Calculate sum of word vector in window
//                        for (long long k = 0; k < layer1Size; k++) neu1[k] += syn0[k + index * layer1Size];
                    for (long long k = 0; k < w2v.layer1Size; k++)
                    { neu1[k] += w2v.syn0New[l1][k]; }
                    cw++;
                }
            }

            if (cw)
            {
                for (long long i = 0; i < w2v.layer1Size; i++)
                {
                    neu1[i] /= cw;
                }

                //如果采用分层softmax优化
                //根据Huffman树上从根节点到当前词的叶节点的路径，遍历所有经过的中间节点
                if (w2v.hs)
                {
                    for (size_t i = 0; i < word->second.code.size(); i++)
                    {
                        double f = 0;
                        l2 = (long long) word->second.point[i];
                        for (int j = 0; j < w2v.layer1Size; j++)
                        {
                            f += w2v.syn1New[l2][j] * neu1[j];
                        }
                        //检测f有没有超出Sigmoid函数表的范围
                        if (f <= -w2v.MAX_EXP)
                        { continue; }
                        else if (f >= w2v.MAX_EXP)
                        {
                            continue;
                            //如果没有超出范围则对f进行Sigmoid变换
                        }
                        else
                        { f = w2v.expTable[(int) ((f + w2v.MAX_EXP) * (w2v.EXP_TABLE_SIZE / w2v.MAX_EXP / 2))]; }

                        double g = (1 - word->second.code[i] - f) * w2v.alpha;
                        for (int j = 0; j < w2v.layer1Size; j++)
                        {
                            neu1e[j] += g * w2v.syn1New[l2][j];
                        }
                        for (int j = 0; j < w2v.layer1Size; j++)
                        {
                            w2v.syn1New[l2][j] += g * neu1[j];
                        }
                    }
                }

                if (w2v.negative > 0)
                {
                    uniform_int_distribution<unsigned long> dist2(0, w2v.table.size() - 1);
                    for (int i = 0; i < w2v.negative + 1; i++)
                    {
                        unordered_map<size_t, WordInVocab>::iterator target;
                        long long label;

                        if (i == 0)
                        {
                            target = word;
                            label = 1;
                        }
                        else
                        {
                            auto nextRandom = dist2(mt);
                            target = w2v.table[nextRandom];
                            if (target == word)
                            { continue; }
                            label = 0;
                        }

                        l2 = (long long) target->second.currentNode;
                        double f = 0;
                        for (int j = 0; j < w2v.layer1Size; j++)
                        {
                            f += neu1[j] * w2v.syn1negNew[l2][j];
                        }
                        double g;

                        if (f > w2v.MAX_EXP)
                        { g = (label - 1) * w2v.alpha; }
                        else if (f < -w2v.MAX_EXP)
                        {
                            g = (label - 0) * w2v.alpha;
                            //g = (label - f)*alpha
                        }
                        else
                        {
                            g = (label -
                                 w2v.expTable[(int) ((f + w2v.MAX_EXP) * (w2v.EXP_TABLE_SIZE / w2v.MAX_EXP / 2))]) *
                                w2v.alpha;
                        }
                        //用辅助向量和g更新累计误差
                        for (int j = 0; j < w2v.layer1Size; j++)
                        { neu1e[j] += g * w2v.syn1negNew[l2][j]; }
                        //用输入向量和g更新辅助向量
                        for (int j = 0; j < w2v.layer1Size; j++)
                        { w2v.syn1negNew[l2][j] += g * neu1[j]; }

                    }

                    for (long long a = realWindowSize; a < w2v.window * 2 + 1 - realWindowSize; a++)
                    {
                        if (a != w2v.window)
                        {
                            long long c = sentencePosition - w2v.window + a;
                            if (c < 0)
                            { continue; }
                            if (c >= sentence.size())
                            { continue; }
                            lastWord = sentence[c];
                            if (lastWord == w2v.vocab.getVocabEndIter())
                            { continue; }

                            auto l = (long long) lastWord->second.currentNode;

                            for (c = 0; c < w2v.layer1Size; c++)
                            { w2v.syn0New[l][c] += neu1e[c]; }
                        }
                    }


                }


            }


        }
        sentencePosition++;
        //处理完一句句子后，将句子长度置为零，进入循环，重新读取句子并进行逐词计算
        if (sentencePosition >= sentence.size())
        {
            sentencePosition = 0;
            sentence.erase(sentence.begin(), sentence.end());
            continue;
        }


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

    auto v1Iter = v1.begin();
    auto v2Iter = v2.begin();
    while (v1Iter != v1.end())
    {
        c += (*v1Iter * *v2Iter);
        v1Iter++;
        v2Iter++;
    }
    return c;
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

    if (w2v.negative > 0)
    { w2v.initUnigramTable(); }
    w2v.start = clock();

    vector<thread> threadPool;

//    for (int i = 0; i < w2v.numThreads; i++)
//    {
//        threadPool.emplace_back(trainModelThread, ref(w2v), i);
//    }
//
//for(int i=0;i<w2v.numThreads;i++)
//{
//    threadPool[i].join();
//}

    trainModelThread(w2v, 0);
    cout << "Hello world" << endl;

    ofstream outFile;
    outFile.open("result", std::ofstream::out);
    if (w2v.classes == 0)
    {
        outFile << w2v.vocab.getVocabSize() << "\t" << w2v.layer1Size << endl;
        for (auto it:w2v.vocab.vocab)
        {
            outFile << it.second.word << "\t";
            auto index = (long long) it.second.currentNode;
            for (int j = 0; j < w2v.layer1Size; j++)
            {
                outFile << w2v.syn0New[index][j] << " ";
            }
            outFile << endl;
        }
        outFile << endl;
    }

    vector<string> stringList = {"Paris", "Beijing", "China", "French"};
    auto iter = w2v.getVector(stringList[0]);
//    cout<< (iter == w2v.syn0New.end())<<endl;
//    printVector(iter->second);

    auto diff1 = getDifference(w2v.getVector(stringList[0])->second, w2v.getVector(stringList[3])->second);
    auto diff2 = getDifference(w2v.getVector(stringList[1])->second, w2v.getVector(stringList[2])->second);
    cout << "Product: " << getDotProduct(diff1, diff2) << endl;
//    cout<<inner_product(diff1.begin(),diff1.end(),diff2.begin(),0)<<endl;

}

int main()
{
    Word2Vec word2Vec("test");
    word2Vec.initNet();
    trainModel(word2Vec);
//    const int EXP_TABLE_SIZE=1000;
//    vector<double>expTable;
//    const int MAX_EXP = 6;



//    v.readVocab();
    return 0;
}




