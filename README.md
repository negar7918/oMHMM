__oMHMM (Orthogonal Mixture of Hidden Markov Models)__

This is the source code of the following paper:

N. Safinianaini and C. P. E. de Souza and H. Boström and J. Lagergren, 
"Orthogonal Mixture of Hidden Markov Models" 2020 ECML PKDD 
(https://ecmlpkdd2020.net/programme/accepted/)

The implementation is based on the standard EM for MHMM implementation from this paper (we disabled the sparsity feature):

Spamhmm: Sparse mixture of hidden markov models for graph connected entities.
2019 International Joint Conference on Neural Networks(IJCNN)
pp. 1–10 (2019)

------------------------------------------------------------------------------------------------------------------------------

**Datasets**
- digits: “pen-based recognition of hand- written digits” dataset in the UCI machine learning repository.
- biology: from the NCBI Sequence Read Archive (SRA) under accession number SRP074289; for pre-processing see readme in directory tests/biology. 
- movements: Libras movement dataset from the UCI machine learning repository. 

------------------------------------------------------------------------------------------------------------------------------

**Required Softwares** 

Python 3.6.2

hmmlearn 0.2.1

cvxpy 1.0.21

numpy 1.16.2

scikit-learn 0.19.1

scipy 1.1.0
