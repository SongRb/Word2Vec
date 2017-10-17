#include <iostream>

#include<cmath>
#include <fstream>
#include <functional>
#include <map>
#include <unordered_map>
#include <queue>
#include <iterator>
#include <sstream>

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
    long long index;
    BinaryTreeNode *currentNode;
    // point is to identify vector
    vector<BinaryTreeNode *> point;
    string word, code = "";
};


class Vocabulary
{
private:
    int minCount;
    BinaryTreeNode *root;
    vector<BinaryTreeNode *> midNodeList;
    string trainFileName;
public:
    Vocabulary(string fileName) : trainFileName(fileName), root(nullptr)
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
            tmp.index = static_cast<long long int>(vocab.size());
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
                auto index = it->second.index;
                vocab.erase(it);
                for (auto &it2:vocab)
                {
                    if (it2.second.index > index)
                    {
                        it2.second.index--;
                    }
                }
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
        ofs.open("vocab", std::ofstream::out | std::ofstream::out);
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
    long long layer1Size = 100;
    Vocabulary vocab;
    string trainFile;
    long long trainFileSize = 0;
    vector<unordered_map<size_t, WordInVocab>::iterator> table;
    const int table_size = static_cast<const int>(1e8);

    int numThreads, window, cbow, hs = 0;
    long long wordCountActual;

    vector<double> syn0, syn1, syn1neg;
    vector<double> expTable;

    const int EXP_TABLE_SIZE = 1000;
    const int MAX_EXP = 6;

    unordered_map<long long, vector<double>> syn0New, syn1New, syn1negNew;


    double alpha = 0.025, starting_alpha, sample = 1e-3;

    int classes = 1;

    long long train_words = 0, iter = 5, label;

    int negative;

    int debugMode;

    clock_t start;
public:
    Word2Vec(const string &trainFileName) : trainFile(trainFileName), vocab(trainFileName)
    {
        vocab.learnFromTrainFile();
        vocab.createBinaryTree();
        negative = 1;
        for (int i = 0; i < EXP_TABLE_SIZE; i++)
        {
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
            table[a] = it;
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
            for (long long b = 0; b < layer1Size; b++)
            {
                double nn = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                nn = nn - 0.5;
                tmp.push_back(nn);
            }
            syn0New.insert(make_pair((long long) n.second.currentNode, tmp));
        }
        if (negative > 0)
        {
            syn1neg.resize(0);
            syn1neg.resize(static_cast<unsigned long>(vocab_size * layer1Size), 0);
            syn1negNew = syn0New;
        }

//        for (long long a = 0; a < vocab_size; a++)
//            for (long long b = 0; b < layer1Size; b++)
//            {
//                next_random = next_random * (unsigned long long) 25214903917 + 11;
//                syn0[a * layer1Size + b] = (((next_random & 0xFFFF) / (double) 65536) - 0.5) / layer1Size;
//            }


        cout << syn1negNew.size() << endl;
        cout << syn0New.size() << endl;
    }

    void *trainModelThread(void *id)
    {
        vector<double> neu1(static_cast<unsigned long>(layer1Size));
        vector<double> neu1e(static_cast<unsigned long>(layer1Size));
        ifstream inFile(trainFile, ifstream::binary);
        inFile.seekg(trainFileSize / (long long) numThreads * (long long) id);

        long long wordCount = 0, lastWordCount = 0;

        vector<unordered_map<size_t, WordInVocab>::iterator> sen;
        unordered_map<size_t, WordInVocab>::iterator lastWord;
        long long l1, l2;
        long long sentencePosition = 0;

        long long localIter = iter;

        while (true)
        {
            if (wordCount - lastWordCount > 10000)
            {
                wordCountActual += wordCount - lastWordCount;
                lastWordCount = wordCount;
                if ((debugMode > 1))
                {
                    clock_t now = clock();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                           wordCountActual / (double) (iter * train_words + 1) * 100,
                           wordCountActual / ((double) (now - start + 1) / (double) CLOCKS_PER_SEC * 1000));
                    fflush(stdout);
                }
                // Slightly decreases alpha.
                alpha = starting_alpha * (1 - wordCountActual / (double) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001)
                {
                    alpha = starting_alpha * 0.0001;
                }
            }
            // Build a sentence of word
            if (sen.empty())
            {
                string line;
                getline(inFile, line);
                std::stringstream ss(line);
                std::istream_iterator<std::string> begin(ss);
                std::istream_iterator<std::string> end;
                std::vector<std::string> senStringVec(begin, end);
                for (auto word:senStringVec)
                {
                    sen.push_back(vocab.getVocabIter(word));
                }

                wordCount += sen.size();
                // TODO Sample words
                sentencePosition = 0;
            }


            if (inFile.eof() || (wordCount > train_words / numThreads))
            {
                wordCountActual += wordCount - lastWordCount;
                localIter--;
                if (localIter == 0)
                { break; }
                wordCount = 0;
                lastWordCount = 0;
                inFile.seekg(trainFileSize / (long long) numThreads * (long long) id);
            }

            auto word = sen[sentencePosition];
            if (word == vocab.getVocabEndIter())
            { continue; }

            fill(neu1.begin(), neu1.end(), 0);
            fill(neu1e.begin(), neu1e.end(), 0);

            int realWindowSize = rand() % window;

            if (cbow)
            {
                long long cw = 0;
                for (long long i = realWindowSize; i < window * 2 + 1 - realWindowSize; i++)
                {
                    if (i != window)
                    {
                        // TODO Check available of unmatched type for window and sentencePosition
                        long long j = sentencePosition - window + i;
                        if (j < 0 || j >= sen.size())
                        { continue; }
                        lastWord = sen[j];
                        if (lastWord == vocab.getVocabEndIter())
                        { continue; }
                        l1 = (long long) lastWord->second.currentNode;
                        auto index = lastWord->second.index;

                        // Calculate sum of word vector in window
//                        for (long long k = 0; k < layer1Size; k++) neu1[k] += syn0[k + index * layer1Size];
                        for (long long k = 0; k < layer1Size; k++)
                        { neu1[k] += syn0New[l1][k]; }
                        cw++;
                    }
                }

                if (cw)
                {
                    for (long long i = 0; i < layer1Size; i++)
                    {
                        neu1[i] /= cw;
                    }

                    if (hs)
                    {
                        for (size_t i = 0; i < word->second.code.size(); i++)
                        {
                            double f = 0;
                            l2 = (long long) word->second.point[i];
                            for (int j = 0; j < layer1Size; j++)
                            {
                                f += syn1New[l2][j] * neu1[j];
                            }
                            //检测f有没有超出Sigmoid函数表的范围
                            if (f <= -MAX_EXP)
                            { continue; }
                            else if (f >= MAX_EXP)
                            {
                                continue;
                                //如果没有超出范围则对f进行Sigmoid变换
                            }
                            else
                            { f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]; }

                            double g = (1 - word.second.code[i] - f) * alpha;
                            for (int j = 0; j < layer1Size; j++)
                            {
                                neu1e[j] += g * syn1New[l2][j];
                            }
                            for (int j = 0; j < layer1Size; j++)
                            {
                                syn1New[l2][j] += g * neu1[j];
                            }
                        }
                    }

                    if (negative > 0)
                    {
                        for (int i = 0; i < negative + 1; i++)
                        {
                            unordered_map<size_t, WordInVocab>::iterator target;

                            if (i == 0)
                            {
                                target = word;
                                label = 0;
                            }
                            else
                            {
                                auto nextRandom = static_cast<int>((rand() >> 16) % table.size());
                                target = table[nextRandom];
                                if (target == word)
                                { continue; }
                                label = 0;
                            }

                            l2 = (long long) target->second.currentNode;
                            double f = 0;
                            for (int j = 0; j < layer1Size; j++)
                            {
                                f += neu1[j] * syn1negNew[l2][j];
                            }
                            double g;

                            if (f > MAX_EXP)
                            { g = (label - 1) * alpha; }
                            else if (f < -MAX_EXP)
                            {
                                g = (label - 0) * alpha;
                                //g = (label - f)*alpha
                            }
                            else
                            { g = (label - expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha; }
                            //用辅助向量和g更新累计误差
                            for (int j = 0; j < layer1Size; j++)
                            { neu1e[j] += g * syn1negNew[l2][j]; }
                            //用输入向量和g更新辅助向量
                            for (int j = 0; j < layer1Size; j++)
                            { syn1negNew[l2][j] += g * neu1[j]; }

                        }

                        for (long long a = realWindowSize; a < window * 2 + 1 - realWindowSize; a++)
                        {
                            if (a != window)
                            {
                                long long c = sentencePosition - window + a;
                                if (c < 0)
                                { continue; }
                                if (c >= sen.size())
                                { continue; }
                                lastWord = sen[c];
                                if (lastWord == vocab.getVocabEndIter())
                                { continue; }

                                auto l = (long long) lastWord->second.currentNode;

                                for (c = 0; c < layer1Size; c++)
                                { syn0New[l][c] += neu1e[c]; }
                            }
                        }


                    }


                }


            }


        }


    }


    void trainModel()
    {
        vector<pthread_t> pt(static_cast<unsigned long>(numThreads));
        starting_alpha = alpha;

        initNet();

        if (negative > 0)
        { initUnigramTable(); }
        start = clock();
        for (int i = 0; i < numThreads; i++)
        {
            pthread_create(&pt[i], nullptr, trainModelThread, (void *) i);
        }
        for (int i = 0; i < numThreads; i++)
        {
            pthread_join(pt[i], nullptr);
        }

        ofstream outFile("out");
        if (classes == 0)
        {
            outFile << vocab.getVocabSize() << "\t" << layer1Size << endl;
            for (auto it:vocab.vocab)
            {
                outFile << it.second.word << "\t";
                auto index = (long long) it.second.currentNode;
                for (int j = 0; j < layer1Size; j++)
                {
                    outFile << syn0New[index][j] << " ";
                }
            }
            outFile << endl;

        }


    }
};

int main()
{
    Word2Vec word2Vec("test");
    word2Vec.initNet();
    word2Vec.trainModel();
//    v.readVocab();
    return 0;
}




