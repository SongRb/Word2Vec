# Word2Vec
C++ Implementation of word2vec  
This project aims to implement Mikolov, Tomas, et al. 
"Distributed representations of words and phrases and their compositionality." 
*Advances in neural information processing systems.* 2013. and follows original C implementation from [dav/word2vec](https://github.com/dav/word2vec)  
Modern C++ 11 standard will be followed in this program to improve readability and speed.

To start, just clone the project and run `cmake && make && ./Word2Vec` under root directory. To test the result, you can put a file named `test1` which contains two sentences like   
```
a king is a man
a queen is a woman
```  
Then you can view the dot product of `vec('king') - vec ('man')` and `vec('queen') - vec ('woman')`, which is a small number.
