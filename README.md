# Word2Vec
C++ Implementation of word2vec  
This project aims to implement Mikolov, Tomas, et al. 
"Distributed representations of words and phrases and their compositionality." 
*Advances in neural information processing systems.* 2013. and follows original C implementation from [dav/word2vec](https://github.com/dav/word2vec)  
Modern C++ 11 standard will be followed in this program to improve readability and speed.

To start, just clone the project and run `cmake && make && ./Word2Vec` under root directory. 
The accepted command line option is:
```
-f Filename to be trained
-m Debug mode to control output
-t Number of training threads
-i Number of iteration
-d Word embedding size
-hs Hierarchical softmax
-ns Negative sampling
```
The training corpus should be a plain text file containing lines of sentence. After training, a file named `result.txt` are saved which contains word and its corresponding vector.

The fulling training process are tested both on Ubuntu 16.04 and Windows 10, feel free to use Visual Studio to analyze the code!
